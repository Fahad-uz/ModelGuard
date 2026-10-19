"""Static chat-template extractor and safety analyzer without template rendering."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import jinja2
    from jinja2 import nodes as jnodes
    HAVE_JINJA2 = True
except ImportError:  # pragma: no cover
    jinja2 = None
    jnodes = None
    HAVE_JINJA2 = False

from modelguard import limits
from modelguard.analyzers.trusted_templates import compare_template_with_references
from modelguard.models.results import Finding, ParsedModel, Severity

logger = logging.getLogger(__name__)

# Conservative template size and recursion ceilings (Slide 8, 16)
MAX_TEMPLATE_BYTES = getattr(limits, "MAX_TEMPLATE_BYTES", 256 * 1024)
MAX_TEMPLATE_AST_DEPTH = getattr(limits, "MAX_TEMPLATE_AST_DEPTH", 50)


# Backdoor phrases and suspicious prompt overrides
SUSPICIOUS_PROMPT_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(?:safety|guardrails|filters)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+in\s+developer\s+mode\b", re.IGNORECASE),
    re.compile(r"\bsecretly\s+(?:exfiltrate|send|leak)\b", re.IGNORECASE),
    re.compile(r"\bhidden\s+backdoor\b", re.IGNORECASE),
    re.compile(r"\boverride\s+system\s+prompt\b", re.IGNORECASE),
]

# Dunder pattern for regex matching
DUNDER_PATTERN = re.compile(r"\b__[a-zA-Z0-9_]+__\b")

# Dangerous builtins/functions that templates must not invoke
DANGEROUS_FUNCTIONS = {
    "eval", "exec", "compile", "open", "os", "sys", "subprocess",
    "popen", "breakpoint", "__import__", "getattr", "setattr", "delattr",
}

# Suspicious sandbox globals
SANDBOX_ESCAPE_GLOBALS = {"lipsum", "cycler", "joiner", "namespace"}


def _finding(
    code: str,
    severity: Severity,
    category: str,
    message: str,
    field: str | None = None,
    evidence: object = None,
    remediation: str | None = None,
) -> Finding:
    return Finding(code, severity, category, message, field, evidence, remediation)


@dataclass(slots=True)
class ExtractedTemplate:
    source: str
    content: str
    name: str = "default"


class TemplateExtractor:
    """Safely extracts chat templates from model metadata or associated package files."""

    def extract(self, model: ParsedModel) -> list[ExtractedTemplate]:
        templates: list[ExtractedTemplate] = []
        metadata = model.metadata

        if isinstance(metadata, dict):
            # 1. GGUF standard metadata key: tokenizer.chat_template
            main_template = metadata.get("tokenizer.chat_template")
            if isinstance(main_template, str) and main_template.strip():
                templates.append(ExtractedTemplate(
                    source="metadata.tokenizer.chat_template",
                    content=main_template,
                    name="default",
                ))

            # 2. Named templates: tokenizer.chat_template.<name>
            for key, val in metadata.items():
                if key.startswith("tokenizer.chat_template.") and isinstance(val, str) and val.strip():
                    name = key.split("tokenizer.chat_template.", 1)[1]
                    templates.append(ExtractedTemplate(
                        source=f"metadata.{key}",
                        content=val,
                        name=name,
                    ))

            # 3. Safetensors / root metadata key: chat_template
            if not templates:
                root_tmpl = metadata.get("chat_template")
                if isinstance(root_tmpl, str) and root_tmpl.strip():
                    templates.append(ExtractedTemplate(
                        source="metadata.chat_template",
                        content=root_tmpl,
                        name="default",
                    ))

        # 4. Associated tokenizer files if available in the model directory
        if not templates and model.path and model.path.parent.is_dir():
            templates.extend(self._extract_from_sibling_files(model.path.parent))

        return templates

    def _extract_from_sibling_files(self, directory: Path) -> list[ExtractedTemplate]:
        results: list[ExtractedTemplate] = []
        # Check tokenizer_config.json
        cfg_path = directory / "tokenizer_config.json"
        if cfg_path.is_file():
            try:
                # Bounded read
                if cfg_path.stat().st_size <= limits.MAX_HEADER_BYTES:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tmpl = data.get("chat_template")
                    if isinstance(tmpl, str) and tmpl.strip():
                        results.append(ExtractedTemplate(
                            source="tokenizer_config.json#chat_template",
                            content=tmpl,
                            name="default",
                        ))
                    elif isinstance(tmpl, list):
                        for item in tmpl:
                            if isinstance(item, dict) and "template" in item:
                                t_name = str(item.get("name", "custom"))
                                t_content = str(item["template"])
                                if t_content.strip():
                                    results.append(ExtractedTemplate(
                                        source=f"tokenizer_config.json#{t_name}",
                                        content=t_content,
                                        name=t_name,
                                    ))
            except Exception as exc:
                logger.debug("Failed to read tokenizer_config.json in %s: %s", directory, exc)

        # Check chat_template.json
        direct_path = directory / "chat_template.json"
        if not results and direct_path.is_file():
            try:
                if direct_path.stat().st_size <= limits.MAX_HEADER_BYTES:
                    with open(direct_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tmpl = data.get("chat_template") if isinstance(data, dict) else None
                    if isinstance(tmpl, str) and tmpl.strip():
                        results.append(ExtractedTemplate(
                            source="chat_template.json#chat_template",
                            content=tmpl,
                            name="default",
                        ))
            except Exception as exc:
                logger.debug("Failed to read chat_template.json in %s: %s", directory, exc)

        return results


class TemplateAnalyzer:
    """Statically checks chat templates without rendering or executing them.

    Keeps code-risk and prompt-content findings separate (Slide 9).
    """

    def __init__(self) -> None:
        self.extractor = TemplateExtractor()

    def analyze(self, model: ParsedModel, *, custom_template: str | None = None) -> list[Finding]:
        findings: list[Finding] = []

        if custom_template is not None:
            templates = [ExtractedTemplate(source="custom", content=custom_template)]
        else:
            templates = self.extractor.extract(model)

        if not templates:
            return findings

        # Record discovered templates in model.extra for reporting
        model.extra["chat_templates"] = [
            {"source": t.source, "name": t.name, "length": len(t.content)}
            for t in templates
        ]

        # Extract declared architecture if present in metadata
        declared_arch = None
        if isinstance(model.metadata, dict):
            arch_val = model.metadata.get("general.architecture")
            if isinstance(arch_val, str):
                declared_arch = arch_val

        for tmpl in templates:
            findings.extend(self._analyze_single_template(tmpl, declared_arch))

        return findings

    def _analyze_single_template(
        self,
        tmpl: ExtractedTemplate,
        declared_arch: str | None = None,
    ) -> list[Finding]:
        findings: list[Finding] = []
        content = tmpl.content
        field_name = tmpl.source

        # 1. Structural / Size checks
        byte_len = len(content.encode("utf-8", errors="replace"))
        if byte_len == 0:
            findings.append(_finding(
                "CT123", "INFO", "template_syntax",
                "Chat template is empty",
                field_name,
            ))
            return findings

        if byte_len > MAX_TEMPLATE_BYTES:
            findings.append(_finding(
                "CT121", "WARNING", "template_syntax",
                f"Chat template size ({byte_len} bytes) exceeds configured limit ({MAX_TEMPLATE_BYTES} bytes)",
                field_name,
                {"size": byte_len, "limit": MAX_TEMPLATE_BYTES},
                "Ensure the chat template is not padded or abused for resource exhaustion.",
            ))

        # 2. Parse AST safely without rendering
        ast: Any = None
        if HAVE_JINJA2:
            try:
                env = jinja2.Environment()
                ast = env.parse(content)
            except jinja2.TemplateSyntaxError as exc:
                findings.append(_finding(
                    "CT120", "ERROR", "template_syntax",
                    f"Malformed Jinja2 syntax: {exc.message}",
                    field_name,
                    {"line": exc.lineno, "message": exc.message},
                    "Fix template syntax to ensure compatibility with standard inference loaders.",
                ))
            except Exception as exc:
                findings.append(_finding(
                    "CT120", "ERROR", "template_syntax",
                    f"Template parser error: {type(exc).__name__}: {exc}",
                    field_name,
                    {"error": str(exc)},
                ))

        # 3. Code-Risk Checks (Category: template_code_risk)
        findings.extend(self._check_code_risks(content, ast, field_name))

        # 4. Prompt-Content Checks (Category: template_prompt_risk)
        findings.extend(self._check_prompt_risks(content, ast, field_name))

        # 5. Reference Comparison Checks (Category: template_reference)
        findings.extend(self._check_reference(content, declared_arch, field_name))

        return findings

    def _check_code_risks(self, content: str, ast: Any, field_name: str) -> list[Finding]:
        findings: list[Finding] = []

        # A. Dunder attribute access or method traversal (CT101 - CRITICAL)
        dunders_found: set[str] = set(DUNDER_PATTERN.findall(content))

        if ast is not None and HAVE_JINJA2:
            # Check Getattr nodes
            for node in ast.find_all(jnodes.Getattr):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    dunders_found.add(node.attr)
            # Check Getitem nodes with constant string keys
            for node in ast.find_all(jnodes.Getitem):
                if (isinstance(node.arg, jnodes.Const)
                        and isinstance(node.arg.value, str)
                        and node.arg.value.startswith("__")
                        and node.arg.value.endswith("__")):
                    dunders_found.add(node.arg.value)

        if dunders_found:
            findings.append(_finding(
                "CT101", "CRITICAL", "template_code_risk",
                "Dunder attribute or method traversal detected in chat template",
                field_name,
                {"dunder_attributes": sorted(dunders_found)},
                "Remove Python dunder accesses; chat templates must not navigate Python object internals.",
            ))

        # B. Disallowed function or system call invocation (CT102 - CRITICAL)
        if ast is not None and HAVE_JINJA2:
            dangerous_calls: set[str] = set()
            for call_node in ast.find_all(jnodes.Call):
                target = call_node.node
                if isinstance(target, jnodes.Name) and target.name in DANGEROUS_FUNCTIONS:
                    dangerous_calls.add(target.name)
                elif isinstance(target, jnodes.Getattr) and target.attr in DANGEROUS_FUNCTIONS:
                    dangerous_calls.add(target.attr)
            if dangerous_calls:
                findings.append(_finding(
                    "CT102", "CRITICAL", "template_code_risk",
                    "Disallowed function or system call invocation in chat template",
                    field_name,
                    {"functions": sorted(dangerous_calls)},
                    "Remove system/code execution calls from template.",
                ))

            # C. Sandbox escape targets (CT104 - CRITICAL)
            escape_targets: set[str] = set()
            for name_node in ast.find_all(jnodes.Name):
                if name_node.name in SANDBOX_ESCAPE_GLOBALS:
                    # Verify if this global is chained with attribute access or calls
                    escape_targets.add(name_node.name)
            if escape_targets and dunders_found:
                findings.append(_finding(
                    "CT104", "CRITICAL", "template_code_risk",
                    "Jinja sandbox escape pattern detected with escape globals",
                    field_name,
                    {"escape_globals": sorted(escape_targets)},
                    "Templates must not access Jinja sandbox escape globals like lipsum or cycler.",
                ))

            # D. File inclusion / external template import (CT105 - ERROR)
            includes: list[str] = []
            for inc in ast.find_all(jnodes.Include):
                if isinstance(inc.template, jnodes.Const) and isinstance(inc.template.value, str):
                    includes.append(inc.template.value)
                else:
                    includes.append("<dynamic>")
            for imp in ast.find_all((jnodes.Import, jnodes.FromImport)):
                if hasattr(imp, "template") and isinstance(imp.template, jnodes.Const):
                    includes.append(str(imp.template.value))
                else:
                    includes.append("<import>")
            if includes:
                findings.append(_finding(
                    "CT105", "ERROR", "template_code_risk",
                    "External template import or file inclusion attempted",
                    field_name,
                    {"targets": includes},
                    "Chat templates must be self-contained; remove include/import tags.",
                ))

            # E. AST nesting depth check (CT122 - WARNING)
            depth = self._max_ast_depth(ast)
            if depth > MAX_TEMPLATE_AST_DEPTH:
                findings.append(_finding(
                    "CT122", "WARNING", "template_syntax",
                    f"Excessive AST nesting depth ({depth} > {MAX_TEMPLATE_AST_DEPTH})",
                    field_name,
                    {"depth": depth, "limit": MAX_TEMPLATE_AST_DEPTH},
                    "Simplify nested Jinja blocks to prevent parser stack overflow.",
                ))

        return findings

    def _check_prompt_risks(self, content: str, ast: Any, field_name: str) -> list[Finding]:
        findings: list[Finding] = []

        # A. System message omission / suppression (CT110 - WARNING)
        # Check if template processes messages and handles roles
        has_messages = "messages" in content
        handles_user = "'user'" in content or '"user"' in content
        handles_assistant = (
            "'assistant'" in content or '"assistant"' in content or
            "'model'" in content or '"model"' in content
        )
        handles_system = "'system'" in content or '"system"' in content

        if has_messages and handles_user and handles_assistant and not handles_system:
            findings.append(_finding(
                "CT110", "WARNING", "template_prompt_risk",
                "System message role is omitted or silently discarded by the template",
                field_name,
                {"roles_handled": ["user", "assistant"], "system_handled": False},
                "Ensure the chat template incorporates system messages to preserve security guardrails.",
            ))

        # B. Hardcoded backdoor instructions or suspicious override (CT111 - WARNING)
        matched_backdoors: list[str] = []
        for pat in SUSPICIOUS_PROMPT_PATTERNS:
            match = pat.search(content)
            if match:
                matched_backdoors.append(match.group(0))

        if matched_backdoors:
            findings.append(_finding(
                "CT111", "WARNING", "template_prompt_risk",
                "Suspicious prompt override or backdoor instructions in chat template text",
                field_name,
                {"matches": matched_backdoors},
                "Inspect template text to ensure it does not covertly inject adversarial instructions.",
            ))

        # C. Generation prompt control check (CT113 - INFO)
        if has_messages and "add_generation_prompt" not in content:
            findings.append(_finding(
                "CT113", "INFO", "template_prompt_risk",
                "Template does not reference 'add_generation_prompt' for generation control",
                field_name,
                remediation="Consider supporting 'add_generation_prompt' for standard generation formatting.",
            ))

        return findings

    def _check_reference(
        self,
        content: str,
        declared_arch: str | None,
        field_name: str,
    ) -> list[Finding]:
        findings: list[Finding] = []
        comp = compare_template_with_references(content, declared_arch)

        if comp.architecture_divergence:
            findings.append(_finding(
                "CT131", "WARNING", "template_reference",
                f"Template diverges from canonical reference for architecture '{declared_arch}'",
                field_name,
                comp.evidence,
                "Verify whether the template divergence is intentional for this model variant.",
            ))
        elif comp.matched_reference is not None:
            findings.append(_finding(
                "CT130", "INFO", "template_reference",
                f"Matches canonical reference template: {comp.matched_reference.name}",
                field_name,
                comp.evidence,
            ))
        else:
            findings.append(_finding(
                "CT132", "INFO", "template_reference",
                "Custom chat template format detected; details preserved for review",
                field_name,
                comp.evidence,
            ))

        return findings

    def _max_ast_depth(self, node: Any, current_depth: int = 1) -> int:
        if not hasattr(node, "iter_child_nodes"):
            return current_depth
        max_d = current_depth
        for child in node.iter_child_nodes():
            d = self._max_ast_depth(child, current_depth + 1)
            if d > max_d:
                max_d = d
        return max_d
