"""Bounded static chat-template inspection, without rendering or compilation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
from typing import Any

try:
    import jinja2
    from jinja2 import nodes as jnodes
    from modelguard.analyzers.template_static import (
        BudgetExceeded, GenerationExtension, StaticResolver, inventory, lex_budget,
    )
    HAVE_JINJA2 = True
except ImportError:
    jinja2 = None
    jnodes = None
    HAVE_JINJA2 = False

from modelguard import limits
from modelguard.analyzers.template_extractor import ExtractedTemplate, TemplateExtractor
from modelguard.analyzers.trusted_templates import compare_template_with_references
from modelguard.models.results import Finding, ParsedModel

MAX_TEMPLATE_BYTES = limits.MAX_TEMPLATE_BYTES
MAX_TEMPLATE_AST_DEPTH = limits.MAX_TEMPLATE_AST_DEPTH

SUSPICIOUS_PROMPT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
        r"\bbypass\s+(?:safety|guardrails|filters)\b",
        r"\byou\s+are\s+now\s+in\s+developer\s+mode\b",
        r"\bsecretly\s+(?:exfiltrate|send|leak)\b",
        r"\bhidden\s+backdoor\b", r"\boverride\s+system\s+prompt\b",
    )
]
DANGEROUS_FUNCTIONS = {
    "eval", "exec", "compile", "open", "popen", "breakpoint", "__import__",
    "getattr", "setattr", "delattr",
}
SYSTEM_MODULES = {"os", "sys", "subprocess", "posix", "builtins", "__builtins__"}
SANDBOX_ESCAPE_GLOBALS = {"lipsum", "cycler", "joiner", "namespace"}


def _finding(code: str, severity: str, category: str, message: str,
             source: str, evidence: object = None, remediation: str | None = None) -> Finding:
    return Finding(code, severity, category, message, source, evidence, remediation)


class TemplateAnalyzer:
    """Own extraction, inspection budgets and coverage for every entry point."""

    def __init__(self) -> None:
        self.extractor = TemplateExtractor()
        self._jinja_env: Any = None
        if HAVE_JINJA2:
            self._jinja_env = jinja2.Environment(extensions=[GenerationExtension, "jinja2.ext.loopcontrols"])

    def analyze(self, model: ParsedModel, *, custom_template: str | None = None,
                profile: str = "auto") -> list[Finding]:
        if custom_template is None:
            templates = self.extractor.extract(model, profile=profile)
        else:
            templates = [ExtractedTemplate("custom", custom_template)]
            model.extra["chat_templates"] = []
            model.extra["template_analysis"] = {
                "status": "pending", "complete": True, "loader_profile": "custom",
                "inspected_sources": [], "skipped_sources": [], "conflicts": [],
                "extraction_findings": [],
            }
        state = model.extra["template_analysis"]
        if model.parser_errors:
            state["complete"] = False
            state["skipped_sources"].append({"source": "container metadata", "reason": "Container parsing was incomplete"})
        findings = [Finding(**item) for item in state.pop("extraction_findings", [])]
        summaries: list[dict[str, object]] = []
        model.extra["chat_templates"] = summaries
        state["dialect"] = "transformers-jinja (generation and loop controls)"
        state["runtime_verified"] = False
        state["limitations"] = [
            "Static indicators do not establish malicious intent or downstream exploitability.",
            "Role-flow, prompt content and format resemblance checks are conservative heuristics.",
        ]
        arch = model.metadata.get("general.architecture") if isinstance(model.metadata, dict) else None
        arch = arch if isinstance(arch, str) else None
        deadline = time.monotonic() + limits.MAX_TEMPLATE_SCAN_SECONDS
        total_bytes = 0
        for index, template in enumerate(templates):
            summary: dict[str, object] = {
                "source": template.source, "name": template.name,
                "length": len(template.content), "effective": template.effective,
            }
            summaries.append(summary)
            byte_len = len(template.content.encode("utf-8", errors="replace"))
            total_bytes += byte_len
            if (index >= limits.MAX_TEMPLATES or byte_len > MAX_TEMPLATE_BYTES
                    or total_bytes > limits.MAX_TEMPLATE_TOTAL_BYTES):
                found = [_finding("CT121", "WARNING", "template_coverage", "Template resource budget exceeded; parsing skipped", template.source)]
                detail = {"status": "skipped", "complete": False}
            elif self._jinja_env is None:
                found = [_finding("CT124", "ERROR", "template_coverage", "Jinja parser unavailable; AST inspection was not performed", template.source)]
                detail = {"status": "unavailable", "complete": False}
            elif time.monotonic() >= deadline:
                found = [_finding("CT125", "ERROR", "template_coverage", "Template scan deadline exceeded; remaining source skipped", template.source)]
                detail = {"status": "skipped", "complete": False}
            else:
                found, detail = self._run_worker(template, arch, min(limits.MAX_TEMPLATE_WORKER_SECONDS, deadline - time.monotonic()))
            summary.update(detail)
            if not summary.get("complete"):
                state["complete"] = False
                state["skipped_sources"].append({"source": template.source, "reason": summary.get("status")})
            for finding in found:
                evidence = dict(finding.evidence) if isinstance(finding.evidence, dict) else {}
                evidence.update({"template_name": template.name, "effective": template.effective,
                                 "runtime_verified": False})
                finding.evidence = evidence
                # Loader selection is context, never a reason to suppress a
                # syntax indicator: applications may override template sources.
                findings.append(finding)
        if not state["complete"]:
            state["status"] = "incomplete"
        elif templates:
            state["status"] = "completed"
        else:
            state["status"] = "absent"
        state["templates_inspected"] = sum(item.get("status") == "completed" for item in summaries)
        state["templates_discovered"] = len(templates)
        return findings

    @staticmethod
    def _run_worker(template: ExtractedTemplate, arch: str | None, timeout: float) -> tuple[list[Finding], dict[str, object]]:
        request = json.dumps({"template": asdict(template), "architecture": arch}, ensure_ascii=True)
        if len(request.encode("utf-8")) > limits.MAX_TEMPLATE_WORKER_OUTPUT_BYTES * 2:
            return [_finding("CT121", "WARNING", "template_coverage", "Template request exceeds worker budget", template.source)], {"status": "skipped", "complete": False}
        environment = dict(os.environ)
        # Isolated Python must never import a model-directory package or
        # sitecustomize.py through cwd/PYTHONPATH. Bootstrap only the already
        # imported, trusted package and dependency locations.
        import markupsafe
        package_root = str(Path(__file__).resolve().parents[2])
        paths = list(dict.fromkeys([package_root, str(Path(jinja2.__file__).resolve().parent.parent),
                                    str(Path(markupsafe.__file__).resolve().parent.parent)]))
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        bootstrap = "import sys,runpy;sys.path[:0]=sys.argv[1:];runpy.run_module('modelguard.template_worker',run_name='__main__')"
        try:
            started = time.monotonic()
            completed = subprocess.run(
                [sys.executable, "-I", "-c", bootstrap, *paths], input=request,
                capture_output=True, text=True, encoding="utf-8", env=environment,
                timeout=max(0.01, timeout), check=False, cwd=package_root,
            )
            if completed.returncode or len(completed.stdout.encode("utf-8")) > limits.MAX_TEMPLATE_WORKER_OUTPUT_BYTES:
                raise ValueError("Template worker did not return a bounded result")
            result = json.loads(completed.stdout)
            result["summary"]["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            return [Finding(**item) for item in result["findings"]], result["summary"]
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            return [_finding("CT125", "ERROR", "template_coverage", "Template worker failed or exceeded its deadline", template.source,
                             {"reason": type(exc).__name__})], {"status": "failed", "complete": False}

    def _inspect_single_template(self, template: ExtractedTemplate, arch: str | None = None) -> tuple[list[Finding], dict[str, object]]:
        """Worker implementation; only bounded syntax inspection occurs here."""
        content, source = template.content, template.source
        summary: dict[str, object] = {
            "status": "completed", "complete": True,
            "sha256": hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
        }
        if len(content.encode("utf-8", errors="replace")) > MAX_TEMPLATE_BYTES:
            return [_finding("CT121", "WARNING", "template_coverage", "Template byte limit exceeded", source)], {"status": "skipped", "complete": False}
        if not content.strip():
            return [_finding("CT123", "INFO", "template_syntax", "Chat template is empty", source)], summary
        try:
            lex_budget(self._jinja_env, content)
            ast = self._jinja_env.parse(content)
            ast_nodes = inventory(ast)
        except (BudgetExceeded, RecursionError, MemoryError) as exc:
            return [_finding("CT122", "WARNING", "template_syntax", f"Template complexity limit reached: {type(exc).__name__}", source)], {"status": "skipped", "complete": False}
        except jinja2.TemplateSyntaxError as exc:
            unsupported = "unknown tag" in exc.message.lower()
            return [_finding("CT126" if unsupported else "CT120", "ERROR", "template_syntax",
                             f"{'Unsupported template dialect' if unsupported else 'Malformed Jinja syntax'}: {exc.message[:240]}",
                             source, {"line": exc.lineno})], {"status": "unsupported" if unsupported else "failed", "complete": False}
        resolver = StaticResolver(ast_nodes)
        summary["ast_nodes"] = len(ast_nodes)
        summary["ast_depth"] = self._max_ast_depth(ast)
        summary["ast_node_types"] = sorted({type(node).__name__ for node in ast_nodes})
        findings = []
        try:
            findings.extend(self._code_risks(ast_nodes, resolver, source))
            comparison = compare_template_with_references(content, arch)
            findings.extend(self._prompt_risks(ast_nodes, resolver, source, comparison))
        except (BudgetExceeded, RecursionError, MemoryError) as exc:
            findings.append(_finding("CT122", "WARNING", "template_syntax", f"Static analysis budget reached: {type(exc).__name__}", source))
            summary.update({"status": "skipped", "complete": False})
            return findings, summary
        summary["reference"] = comparison.evidence
        if comparison.evidence.get("exact_reference"):
            message = "Exact byte match to pinned reference snapshot; model identity and safety are not authenticated"
            code, severity = "CT134", "INFO"
        elif comparison.architecture_divergence:
            message = "Template format differs from a suggested architecture profile; custom variants may be legitimate"
            code, severity = "CT131", "WARNING"
        elif comparison.matched_reference is not None:
            message = f"Resembles format example: {comparison.matched_reference.name}; provenance unverified"
            code, severity = "CT130", "INFO"
        else:
            message = "Custom template format; no verified reference match"
            code, severity = "CT132", "INFO"
        findings.append(_finding(code, severity, "template_reference", message, source, comparison.evidence))
        return findings, summary

    def _analyze_single_template(self, template: ExtractedTemplate, declared_arch: str | None = None) -> list[Finding]:
        """Compatibility helper with the same worker boundary as analyze()."""
        return self._run_worker(template, declared_arch, limits.MAX_TEMPLATE_WORKER_SECONDS)[0]

    @staticmethod
    def _code_risks(ast_nodes: list[Any], resolver: Any, source: str) -> list[Finding]:
        findings = []
        dunders: set[str] = set()
        calls: set[str] = set()
        escape_targets: set[str] = set()
        includes: list[str] = []
        unresolved_lines: set[int] = set()
        for node in ast_nodes:
            paths = resolver.paths(node) if isinstance(node, (jnodes.Getattr, jnodes.Getitem, jnodes.Filter, jnodes.Call)) else set()
            for path in paths:
                attrs = [part for part in path[1:] if part.startswith("__") and part.endswith("__")]
                dunders.update(attrs)
                if attrs and path[0] in SANDBOX_ESCAPE_GLOBALS:
                    escape_targets.add(path[0])
            if isinstance(node, jnodes.Call):
                for path in resolver.paths(node.node):
                    if path[-1] in DANGEROUS_FUNCTIONS or any(part in SYSTEM_MODULES for part in path[:-1]):
                        calls.add(".".join(path))
            if isinstance(node, jnodes.Getitem) and resolver.string(node.arg) is None:
                numeric = isinstance(node.arg, (jnodes.Slice, jnodes.Const)) and (
                    isinstance(node.arg, jnodes.Slice) or type(node.arg.value) is int)
                if not numeric and any(path[0] in SYSTEM_MODULES | SANDBOX_ESCAPE_GLOBALS or any(part.startswith("__") for part in path)
                                       for path in resolver.paths(node.node)):
                    unresolved_lines.add(node.lineno)
            if isinstance(node, jnodes.Filter):
                attribute = None
                if node.name in {"attr", "selectattr", "rejectattr"} and node.args:
                    attribute = resolver.string(node.args[0])
                    if attribute is None:
                        unresolved_lines.add(node.lineno)
                for keyword in node.kwargs:
                    if node.name == "map" and keyword.key == "attribute":
                        attribute = resolver.string(keyword.value)
                        if attribute is None:
                            unresolved_lines.add(node.lineno)
                if attribute and attribute.startswith("__") and attribute.endswith("__"):
                    dunders.add(attribute)
            if isinstance(node, (jnodes.Include, jnodes.Import, jnodes.FromImport, jnodes.Extends)):
                includes.append((resolver.string(node.template) or "<dynamic>")[:240])
        if dunders:
            findings.append(_finding("CT101", "CRITICAL", "template_code_risk", "Python object-internals access in a template expression", source,
                                     {"dunder_attributes": sorted(dunders), "confidence": "static indicator"}, "Remove object-internals traversal; verify the actual loader sandbox separately."))
        if calls:
            findings.append(_finding("CT102", "CRITICAL", "template_code_risk", "Potential code, process or file access in a template call", source,
                                     {"functions": sorted(calls)}, "Remove these calls from the template."))
        if escape_targets:
            findings.append(_finding("CT104", "CRITICAL", "template_code_risk", "Object-internals traversal through a Jinja global", source,
                                     {"escape_globals": sorted(escape_targets)}, "Remove the traversal; this finding does not prove a sandbox escape."))
        if includes:
            findings.append(_finding("CT105", "ERROR", "template_code_risk", "External template inclusion, import or inheritance", source,
                                     {"targets": includes[:32]}, "Use a self-contained chat template."))
        if unresolved_lines:
            findings.append(_finding("CT103", "WARNING", "template_coverage", "Dynamic attribute names could not be statically resolved", source,
                                     {"lines": sorted(unresolved_lines)[:32], "confidence": "unresolved"}))
        return findings

    @staticmethod
    def _prompt_risks(ast_nodes: list[Any], resolver: Any, source: str, comparison: Any) -> list[Finding]:
        findings = []
        names = {node.name for node in ast_nodes if isinstance(node, jnodes.Name)}
        roles = resolver.role_comparisons(ast_nodes)
        rejects_system = False
        drops_system = False
        for node in ast_nodes:
            if not isinstance(node, jnodes.If):
                continue
            test_nodes = inventory(node.test)
            if "system" not in resolver.role_comparisons(test_nodes):
                continue
            body_nodes = [child for statement in node.body for child in inventory(statement)]
            rejects = any(isinstance(child, jnodes.Call) and any(path[-1] == "raise_exception" for path in resolver.paths(child.node)) for child in body_nodes)
            rejects_system |= rejects
            if not rejects and (not node.body or any(isinstance(child, jnodes.Continue) for child in body_nodes)):
                drops_system = True
        reference = comparison.matched_reference
        lacks_system_by_design = reference is not None and "system" not in reference.expected_roles
        if drops_system or ("messages" in names and {"user", "assistant"}.issubset(roles) and "system" not in roles):
            findings.append(_finding("CT110", "INFO" if lacks_system_by_design else "WARNING", "template_prompt_risk",
                                     "Possible system-role filtering; intentional format restrictions require review", source,
                                     {"roles_compared": sorted(roles), "confidence": "heuristic", "format_supports_system": not lacks_system_by_design}))
        elif rejects_system:
            findings.append(_finding("CT110", "INFO", "template_prompt_risk", "Template explicitly rejects system-role input", source,
                                     {"confidence": "static indicator"}))
        matches = []
        controls = []
        for line, text in resolver.output_strings(ast_nodes):
            normalized = unicodedata.normalize("NFKC", text)
            normalized = "".join(char for char in normalized if unicodedata.category(char) != "Cf")
            for pattern in SUSPICIOUS_PROMPT_PATTERNS:
                match = pattern.search(normalized)
                if match and len(matches) < 16:
                    matches.append({"line": line, "match": match.group(0)[:200], "normalized": normalized != text,
                                    "original_excerpt": text[:240]})
            if len(controls) < 32:
                controls.extend({"line": line, "codepoint": f"U+{ord(char):04X}"} for char in text
                                if unicodedata.category(char) == "Cf")
                controls = controls[:32]
        if matches:
            findings.append(_finding("CT111", "WARNING", "template_prompt_risk", "Suspicious instruction phrase in potential template output; review context", source,
                                     {"matches": matches, "confidence": "heuristic"}))
        if controls:
            findings.append(_finding("CT112", "WARNING", "template_prompt_risk", "Invisible or bidirectional formatting controls in potential template output", source,
                                     {"controls": controls, "confidence": "heuristic"}))
        if "messages" in names and "add_generation_prompt" not in names:
            findings.append(_finding("CT113", "INFO", "template_compatibility", "Template does not reference add_generation_prompt; some formats intentionally omit it", source))
        return findings

    @staticmethod
    def _max_ast_depth(root: Any) -> int:
        maximum = 0
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            maximum = max(maximum, depth)
            stack.extend((child, depth + 1) for child in node.iter_child_nodes())
        return maximum
