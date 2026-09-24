"""Orchestrate static parsing and independent structural analysis."""

from __future__ import annotations

import logging
from pathlib import Path

from modelguard.analyzers.structure_analyzer import StructureAnalyzer
from modelguard.models.results import Finding, ParsedModel, ParserIssue, ScanResult
from modelguard.parsers.base import ParserError, detect_format
from modelguard.parsers.gguf_parser import GGUFParser
from modelguard.parsers.safetensors_parser import SafetensorsParser

logger = logging.getLogger(__name__)


def scan_file(path: str | Path) -> ScanResult:
    """Scan a file without loading tensor contents or executing model data."""
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
        return ScanResult(parsed, StructureAnalyzer().analyze(parsed))

    parser = GGUFParser() if format_name == "GGUF" else SafetensorsParser()
    try:
        parsed = parser.parse(model_path)
    except Exception as exc:
        # This is the single containment boundary for unexpected parser failures.
        logger.exception("Unexpected parser failure while scanning %s", model_path)
        parsed = ParsedModel(model_path, format_name, size,
                             parser_errors=[ParserIssue("MG999", f"Parser failed safely: {type(exc).__name__}: {exc}", "parser")])
    try:
        findings = StructureAnalyzer().analyze(parsed)
    except Exception as exc:
        logger.exception("Unexpected analysis failure while scanning %s", model_path)
        findings = [Finding(issue.code, "ERROR", "parser", issue.message, issue.field, issue.evidence)
                    for issue in parsed.parser_errors]
        findings.append(Finding("MG998", "ERROR", "analyzer", f"Analysis failed safely: {type(exc).__name__}: {exc}"))
    return ScanResult(parsed, findings)
