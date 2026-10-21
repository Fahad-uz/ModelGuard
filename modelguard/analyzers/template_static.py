"""Bounded AST utilities. These inspect syntax and never evaluate templates."""

from __future__ import annotations

import re
from typing import Any

from jinja2 import nodes
from jinja2.ext import Extension

from modelguard import limits


class GenerationExtension(Extension):
    """Parse the Transformers generation block as a scope, without a renderer."""

    tags = {"generation"}

    def parse(self, parser: Any) -> Any:
        line = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body).set_lineno(line)


class BudgetExceeded(ValueError):
    pass


def inventory(root: Any) -> list[Any]:
    """Check depth and node count before any analysis; no recursive walkers."""
    result = []
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limits.MAX_TEMPLATE_AST_DEPTH:
            raise BudgetExceeded("AST depth limit exceeded")
        if len(result) >= limits.MAX_TEMPLATE_AST_NODES:
            raise BudgetExceeded("AST node limit exceeded")
        result.append(node)
        stack.extend((child, depth + 1) for child in reversed(list(node.iter_child_nodes())))
    return result


def lex_budget(environment: Any, content: str) -> None:
    """Reject deep syntax and oversized token streams before recursive parsing."""
    depth = 0
    expression_depth = 0
    block: list[str] | None = None
    opens = {"for", "if", "macro", "call", "filter", "block", "with", "autoescape", "generation"}
    for count, (_, kind, value) in enumerate(environment.lex(content), 1):
        if count > limits.MAX_TEMPLATE_TOKENS:
            raise BudgetExceeded("Template token limit exceeded")
        if kind == "block_begin":
            block = []
        elif kind == "block_end" and block is not None:
            words = [word for word in block if word.strip()]
            if words:
                if words[0] in opens or (words[0] == "set" and "=" not in words):
                    depth += 1
                elif words[0].startswith("end"):
                    depth = max(0, depth - 1)
            block = None
        elif block is not None:
            block.append(value)
        if kind == "operator":
            if value in {"(", "[", "{"}:
                expression_depth += 1
            elif value in {")", "]", "}"}:
                expression_depth = max(0, expression_depth - 1)
        if max(depth, expression_depth) > limits.MAX_TEMPLATE_AST_DEPTH:
            raise BudgetExceeded("Template syntax depth limit exceeded")


class StaticResolver:
    """Conservative union of assignment aliases and bounded constant strings.

    Unknown values stay unknown. Only explicit syntax forms below are handled;
    no eval, compilation, template rendering, or user-provided callable is used.
    """

    def __init__(self, ast_nodes: list[Any]):
        self.aliases: dict[str, list[Any]] = {}
        def bind(target: Any, value: Any) -> None:
            if isinstance(target, nodes.Name):
                self.aliases.setdefault(target.name, []).append(value)
            elif isinstance(target, nodes.Tuple) and isinstance(value, (nodes.Tuple, nodes.List)) and len(target.items) == len(value.items):
                for child, item in zip(target.items, value.items):
                    bind(child, item)
        for node in ast_nodes:
            if isinstance(node, nodes.Assign):
                bind(node.target, node.node)

    def string(self, node: Any, seen: frozenset[str] = frozenset(), depth: int = 0) -> str | None:
        if depth > limits.MAX_TEMPLATE_AST_DEPTH:
            return None
        resolve = lambda item: self.string(item, seen, depth + 1)
        value = None
        if isinstance(node, nodes.Const) and isinstance(node.value, str):
            value = node.value
        elif isinstance(node, nodes.TemplateData):
            value = node.data
        elif isinstance(node, nodes.Name) and node.name not in seen:
            candidates = {self.string(item, seen | {node.name}, depth + 1)
                          for item in self.aliases.get(node.name, [])}
            if len(candidates) == 1:
                value = candidates.pop()
        elif isinstance(node, (nodes.Add, nodes.Concat)):
            parts = node.nodes if isinstance(node, nodes.Concat) else [node.left, node.right]
            resolved = [resolve(item) for item in parts]
            if all(isinstance(item, str) for item in resolved):
                if sum(len(item.encode("utf-8", errors="replace")) for item in resolved) > limits.MAX_TEMPLATE_BYTES:
                    raise BudgetExceeded("Resolved string expansion limit exceeded")
                value = "".join(resolved)
        elif isinstance(node, nodes.Mul):
            base, count = resolve(node.left), node.right
            if base is None:
                base, count = resolve(node.right), node.left
            if base is not None and isinstance(count, nodes.Const) and type(count.value) is int:
                if count.value < 0 or not base:
                    value = ""
                elif len(base.encode("utf-8", errors="replace")) * count.value <= limits.MAX_TEMPLATE_BYTES:
                    value = base * count.value
                else:
                    raise BudgetExceeded("Resolved string repetition limit exceeded")
        elif isinstance(node, nodes.Filter):
            base = resolve(node.node)
            args = [resolve(item) for item in node.args]
            if base is not None:
                if node.name == "lower":
                    value = base.lower()
                elif node.name == "upper":
                    value = base.upper()
                elif node.name == "trim":
                    value = base.strip()
                elif node.name == "replace" and len(args) in {2, 3} and all(isinstance(x, str) for x in args[:2]):
                    # Bound expansion before doing the replacement.
                    old, new = args[:2]
                    if (len(base) + 1) * max(1, len(new)) <= limits.MAX_TEMPLATE_BYTES:
                        value = base.replace(old, new)
                elif node.name == "format":
                    value = self._format(base, node.args, resolve)
            if node.name == "join" and isinstance(node.node, (nodes.List, nodes.Tuple)):
                items = [resolve(item) for item in node.node.items]
                separator = args[0] if args else ""
                if isinstance(separator, str) and all(isinstance(item, str) for item in items):
                    if sum(len(item) for item in items) + len(separator) * len(items) <= limits.MAX_TEMPLATE_BYTES:
                        value = separator.join(items)
        elif isinstance(node, nodes.Mod):
            fmt = resolve(node.left)
            if fmt is not None:
                args = node.right.items if isinstance(node.right, (nodes.Tuple, nodes.List)) else [node.right]
                value = self._format(fmt, args, resolve)
        elif isinstance(node, nodes.Getitem):
            base = resolve(node.node)
            if base is not None and isinstance(node.arg, nodes.Slice):
                parts = [getattr(node.arg, field) for field in ("start", "stop", "step")]
                if all(item is None or (isinstance(item, nodes.Const) and isinstance(item.value, int)) for item in parts):
                    try:
                        value = base[slice(*(None if item is None else item.value for item in parts))]
                    except ValueError:
                        pass
        if isinstance(value, str):
            if len(value.encode("utf-8", errors="replace")) > limits.MAX_TEMPLATE_BYTES:
                raise BudgetExceeded("Resolved string expansion limit exceeded")
            return value
        return None

    @staticmethod
    def _format(fmt: str, args: list[Any], resolve: Any) -> str | None:
        # Width/precision and object formatting are deliberately unsupported.
        if len(fmt) > 8192 or re.search(r"%(?![%scd])", fmt):
            return None
        values = []
        for item in args:
            if isinstance(item, nodes.Const) and type(item.value) is int and 0 <= item.value <= 0x10FFFF:
                values.append(item.value)
            else:
                value = resolve(item)
                if value is None:
                    return None
                values.append(value)
        if len(values) > 128 or sum(len(str(item)) for item in values) + len(fmt) > limits.MAX_TEMPLATE_BYTES:
            return None
        try:
            return fmt % tuple(values)
        except (TypeError, ValueError, OverflowError):
            return None

    def paths(self, node: Any, seen: frozenset[str] = frozenset(), depth: int = 0) -> set[tuple[str, ...]]:
        if depth > limits.MAX_TEMPLATE_AST_DEPTH:
            return set()
        if isinstance(node, nodes.Name):
            paths = {(node.name,)}
            if node.name not in seen:
                for alias in self.aliases.get(node.name, []):
                    paths.update(self.paths(alias, seen | {node.name}, depth + 1))
                    if len(paths) > 32:
                        raise BudgetExceeded("Assignment alias-path budget exceeded")
            return paths
        if isinstance(node, nodes.Call):
            return self.paths(node.node, seen, depth + 1)
        if isinstance(node, (nodes.Getattr, nodes.Getitem, nodes.Filter)):
            if isinstance(node, nodes.Getitem):
                selected = self._indexed_nodes(node.node, node.arg, seen, depth + 1)
                if selected:
                    result = set()
                    for item in selected:
                        result.update(self.paths(item, seen, depth + 1))
                        if len(result) > 32:
                            raise BudgetExceeded("Indexed alias-path budget exceeded")
                    return result
            if isinstance(node, nodes.Getattr):
                attr = node.attr
            elif isinstance(node, nodes.Getitem):
                attr = self.string(node.arg)
            elif node.name == "attr" and node.args:
                attr = self.string(node.args[0])
            else:
                return set()
            if attr is None:
                return set()
            bases = self.paths(node.node, seen, depth + 1) or {("<value>",)}
            return {base + (attr,) for base in bases}
        return set()

    def _indexed_nodes(self, base: Any, index: Any, seen: frozenset[str], depth: int) -> list[Any]:
        if depth > limits.MAX_TEMPLATE_AST_DEPTH:
            raise BudgetExceeded("Indexed alias-depth budget exceeded")
        if isinstance(base, nodes.Name) and base.name not in seen:
            result = []
            for alias in self.aliases.get(base.name, []):
                result.extend(self._indexed_nodes(alias, index, seen | {base.name}, depth + 1))
                if len(result) > 32:
                    raise BudgetExceeded("Indexed alias-path budget exceeded")
            return result
        if isinstance(base, (nodes.Tuple, nodes.List)) and isinstance(index, nodes.Const) and type(index.value) is int:
            if -len(base.items) <= index.value < len(base.items):
                return [base.items[index.value]]
        key = self.string(index)
        if isinstance(base, nodes.Dict) and key is not None:
            return [item.value for item in base.items if self.string(item.key) == key]
        return []

    def role_comparisons(self, ast_nodes: list[Any]) -> set[str]:
        roles: set[str] = set()
        for node in ast_nodes:
            if isinstance(node, nodes.Compare):
                if any(any(path[-1] == "role" for path in self.paths(operand.expr)) for operand in node.ops):
                    role = self.string(node.expr)
                    if role in {"system", "user", "assistant", "model"}:
                        roles.add(role)
                if not any(path[-1] == "role" for path in self.paths(node.expr)):
                    continue
                for operand in node.ops:
                    role = self.string(operand.expr)
                    if role in {"system", "user", "assistant", "model"}:
                        roles.add(role)
                    elif isinstance(operand.expr, (nodes.List, nodes.Tuple)):
                        roles.update(role for item in operand.expr.items
                                     if (role := self.string(item)) in {"system", "user", "assistant", "model"})
        return roles

    def output_strings(self, ast_nodes: list[Any]) -> list[tuple[int, str]]:
        result = []
        total = 0
        pending = ""
        line = 1

        def flush() -> None:
            nonlocal pending
            if pending:
                result.append((line, pending))
                pending = ""

        def walk(statements: list[Any]) -> None:
            nonlocal pending, total, line
            for statement in statements:
                if isinstance(statement, nodes.Output):
                    for item in statement.nodes:
                        value = self.string(item)
                        if value is None:
                            flush()
                            continue
                        if not pending:
                            line = statement.lineno
                        total += len(value.encode("utf-8", errors="replace"))
                        if total > limits.MAX_TEMPLATE_TOTAL_BYTES or len(pending) + len(value) > limits.MAX_TEMPLATE_BYTES:
                            raise BudgetExceeded("Potential output text budget exceeded")
                        pending += value
                elif isinstance(statement, (nodes.Assign, nodes.ExprStmt)):
                    # These statements produce no output and must not split a
                    # deterministic instruction into independent text chunks.
                    continue
                elif isinstance(statement, nodes.Scope):
                    walk(statement.body)
                elif isinstance(statement, nodes.If):
                    conditional([statement, *statement.elif_], statement.else_)
                else:
                    flush()
                    if hasattr(statement, "body") and not isinstance(statement, nodes.AssignBlock):
                        walk(statement.body)
                        flush()
                    if hasattr(statement, "else_"):
                        walk(statement.else_)
                        flush()
        def conditional(branches: list[Any], otherwise: list[Any]) -> None:
            for index, branch in enumerate(branches):
                truth = self._truth(branch.test)
                if truth is True:
                    walk(branch.body)
                    return
                if truth is None:
                    flush()
                    walk(branch.body)
                    flush()
                    conditional(branches[index + 1:], otherwise)
                    flush()
                    return
            walk(otherwise)
        walk(ast_nodes[0].body)
        flush()
        return result

    def _truth(self, node: Any) -> bool | None:
        if isinstance(node, nodes.Const) and type(node.value) in {bool, int, str, type(None)}:
            return bool(node.value)
        if isinstance(node, nodes.Not):
            value = self._truth(node.node)
            return None if value is None else not value
        if isinstance(node, nodes.Compare) and len(node.ops) == 1:
            left, right = self.string(node.expr), self.string(node.ops[0].expr)
            if left is not None and right is not None:
                if node.ops[0].op == "eq":
                    return left == right
                if node.ops[0].op == "ne":
                    return left != right
        return None
