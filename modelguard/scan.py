"""Parse once, then run independent static analyses concurrently."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modelguard.analyzers.structure_analyzer import StructureAnalyzer
from modelguard.models.results import Finding, ParsedModel, ParserIssue, ScanResult
from modelguard.parsers.base import ParserError, detect_format
from modelguard.parsers.gguf_parser import GGUFParser
from modelguard.parsers.safetensors_parser import SafetensorsParser

logger = logging.getLogger(__name__)


def _parser_findings(parsed: ParsedModel) -> list[Finding]:
    return [Finding(issue.code, severity, "parser", issue.message, issue.field, issue.evidence)
            for severity, issues in (("ERROR", parsed.parser_errors), ("WARNING", parsed.parser_warnings))
            for issue in issues]


def _analyze_structure(parsed: ParsedModel) -> list[Finding]:
    scope = parsed.extra["analysis_scope"]
    try:
        findings = StructureAnalyzer().analyze(parsed)
        scope["structure_analysis"] = "completed"
        return findings
    except Exception as exc:
        logger.exception("Unexpected analysis failure while scanning %s", parsed.path)
        scope["structure_analysis"] = "failed"
        return _parser_findings(parsed) + [Finding(
            "MG998", "ERROR", "analyzer", f"Analysis failed safely: {type(exc).__name__}: {exc}")]


def _analyze_templates(parsed: ParsedModel, profile: str) -> list[Finding]:
    try:
        from modelguard.analyzers.template_analyzer import TemplateAnalyzer
        return TemplateAnalyzer().analyze(parsed, profile=profile)
    except Exception as exc:
        logger.exception("Unexpected template analysis failure while scanning %s", parsed.path)
        parsed.extra["template_analysis"] = {"status": "failed", "complete": False,
                                             "reason": "Unexpected template analysis failure"}
        return [Finding("CT999", "ERROR", "template_analyzer",
                        f"Template analysis failed safely: {type(exc).__name__}: {exc}")]


def _scan_scope(template_only: bool, *, runnable: bool = True) -> dict[str, object]:
    return {"mode": "template-only" if template_only else "full",
            "execution": ("single" if template_only else "concurrent") if runnable else "not_run",
            "structure_analysis": "not_requested" if template_only else "not_run"}


def scan_file(path: str | Path, *, template_profile: str = "auto",
              template_only: bool = False) -> ScanResult:
    """Parse a bounded header, then inspect structure and templates in parallel.

    With template_only=True, retain parser diagnostics but skip StructureAnalyzer.
    Tensor contents and template expressions are never loaded or executed.
    """
    model_path = Path(path)
    size = 0
    try:
        size = model_path.stat().st_size
        format_name = detect_format(model_path)
    except (OSError, ParserError) as exc:
        issue = (ParserIssue(exc.code, str(exc), exc.field, exc.evidence)
                 if isinstance(exc, ParserError)
                 else ParserIssue("MG002", f"Cannot read model file: {exc}", "path"))
        parsed = ParsedModel(model_path, "UNKNOWN", size, parser_errors=[issue])
        parsed.extra["analysis_scope"] = _scan_scope(template_only, runnable=False)
        parsed.extra["template_analysis"] = {"status": "not_run", "complete": False,
                                              "reason": "Container could not be opened or identified"}
        return ScanResult(parsed, _parser_findings(parsed))

    parser = GGUFParser() if format_name == "GGUF" else SafetensorsParser()
    try:
        parsed = parser.parse(model_path)
    except Exception as exc:
        # This is the single containment boundary for unexpected parser failures.
        logger.exception("Unexpected parser failure while scanning %s", model_path)
        parsed = ParsedModel(model_path, format_name, size,
                             parser_errors=[ParserIssue("MG999", f"Parser failed safely: {type(exc).__name__}: {exc}", "parser")])
    parsed.extra["analysis_scope"] = _scan_scope(template_only)
    if template_only:
        return ScanResult(parsed, _parser_findings(parsed) + _analyze_templates(parsed, template_profile))

    # StructureAnalyzer reads container facts; TemplateAnalyzer writes only its
    # own coverage/summary keys. Neither changes shared metadata or tensors.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="modelguard") as executor:
        structure = executor.submit(_analyze_structure, parsed)
        templates = executor.submit(_analyze_templates, parsed, template_profile)
        # Keep report order deterministic regardless of which analyzer finishes first.
        findings = structure.result() + templates.result()

    return ScanResult(parsed, findings)


