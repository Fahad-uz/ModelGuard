"""Present normalized scan results without reading the model again."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from modelguard.models.results import Finding, ParserIssue, ScanResult, TensorDescriptor


_SEVERITIES = ("CRITICAL", "ERROR", "WARNING", "INFO")


def _jsonable(value: object, depth: int = 0) -> object:
    """Keep parser evidence JSON serializable, including malformed input values."""
    if depth > 32:
        return "<nested value omitted>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    return str(value)


def _issue_dict(issue: ParserIssue) -> dict[str, object]:
    return {
        "code": issue.code,
        "message": issue.message,
        "field": issue.field,
        "evidence": _jsonable(issue.evidence),
    }


def _tensor_dict(tensor: TensorDescriptor) -> dict[str, object]:
    return {
        "name": tensor.name,
        "dtype": _jsonable(tensor.dtype),
        "shape": _jsonable(tensor.shape),
        "data_offsets": _jsonable(tensor.data_offsets),
        "ggml_type": tensor.ggml_type,
        "descriptor_offset": tensor.descriptor_offset,
    }


def _finding_dict(finding: Finding) -> dict[str, object]:
    return {
        "id": finding.id,
        "severity": finding.severity,
        "category": finding.category,
        "message": finding.message,
        "field": finding.field,
        "evidence": _jsonable(finding.evidence),
        "remediation": finding.remediation,
    }


def report_dict(result: ScanResult) -> dict[str, object]:
    """Return the complete scan as a JSON-ready dictionary."""
    parsed = result.parsed
    severity_counts = {severity: 0 for severity in _SEVERITIES}
    for finding in result.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
    parser_status = (
        "ERROR" if parsed.parser_errors else
        "WARNING" if parsed.parser_warnings else
        "OK"
    )
    scope = parsed.extra.get("analysis_scope", {})
    template_only = scope.get("mode") == "template-only"
    return {
        "file": {
            "path": str(parsed.path),
            "name": parsed.path.name,
            "size_bytes": parsed.file_size,
            "format": parsed.format,
        },
        "parser_status": parser_status,
        "template_analysis": _jsonable(parsed.extra.get("template_analysis", {"status": "not_run", "complete": False})),
        "analysis_scope": {"runtime_verified": False, "model_identity_authenticated": False,
                           "mode": scope.get("mode", "unspecified"),
                           "execution": scope.get("execution", "unspecified"),
                           "structure_analysis": scope.get("structure_analysis", "not_recorded"),
                           "description": ("Static chat-template indicators; container parsing required; structural analysis not requested"
                                           if template_only else
                                           "Static container and template indicators; no inference or exploit validation")},
        "counts": {
            "metadata": len(parsed.metadata),
            "tensors": len(parsed.tensors),
            "findings": len(result.findings),
            "parser_warnings": len(parsed.parser_warnings),
            "parser_errors": len(parsed.parser_errors),
            "by_severity": severity_counts,
        },
        "parsed": {
            "format": parsed.format,
            "file_size": parsed.file_size,
            "format_version": parsed.format_version,
            "header_length": parsed.header_length,
            "data_start": parsed.data_start,
            "declared_tensor_count": parsed.declared_tensor_count,
            "declared_metadata_count": parsed.declared_metadata_count,
            "metadata": _jsonable(parsed.metadata),
            "metadata_types": _jsonable(parsed.metadata_types),
            "tensors": [_tensor_dict(tensor) for tensor in parsed.tensors],
            "parser_warnings": [_issue_dict(issue) for issue in parsed.parser_warnings],
            "parser_errors": [_issue_dict(issue) for issue in parsed.parser_errors],
            "extra": _jsonable(parsed.extra),
        },
        "findings": [_finding_dict(finding) for finding in result.findings],
    }


def _display(value: object) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _terminal(value: object) -> str:
    # JSON escaping keeps model-provided control characters out of terminal output.
    return json.dumps(_display(value), ensure_ascii=True)[1:-1]


def render_terminal(result: ScanResult) -> str:
    """Produce an organized, viewer-friendly, control-character-safe terminal report."""
    report = report_dict(result)
    parsed = result.parsed
    counts = report["counts"]
    assert isinstance(counts, dict)
    by_sev = counts.get("by_severity", {})

    crit_count = by_sev.get("CRITICAL", 0)
    err_count = by_sev.get("ERROR", 0)
    warn_count = by_sev.get("WARNING", 0)
    info_count = by_sev.get("INFO", 0)

    if crit_count > 0:
        verdict = "STATIC CODE-RISK INDICATORS (Loader exploitability not verified)"
    elif err_count > 0:
        verdict = "ERRORS REPORTED (Review findings and analysis coverage)"
    elif warn_count > 0:
        verdict = "WARNING (Anomalies or prompt divergences detected)"
    elif report["parser_status"] == "ERROR":
        verdict = "PARSER FAILED (Corrupted container header)"
    elif report["parser_status"] == "WARNING":
        verdict = "PARSER WARNING (Non-standard container layout)"
    else:
        verdict = "NO ERROR-LEVEL FINDINGS (No safety or authenticity guarantee)"

    coverage = report["template_analysis"]
    scope = report["analysis_scope"]
    template_only = scope["mode"] == "template-only"
    if not coverage.get("complete"):
        verdict += " · TEMPLATE ANALYSIS INCOMPLETE"

    sep = "━" * 72
    thin_sep = "─" * 72

    lines = [
        "ModelGuard review",
        "=" * 17,
        sep,
        ("  MODELGUARD CHAT-TEMPLATE INSPECTION REPORT" if template_only else
         "  MODELGUARD SECURITY & STRUCTURAL INSPECTION REPORT"),
        sep,
        f"File: {_terminal(parsed.path)}",
        f"Format: {_terminal(parsed.format)}" + (f" (v{parsed.format_version})" if parsed.format_version else ""),
        f"Size: {parsed.file_size:,} bytes",
        f"Parser: {report['parser_status']} ({counts['parser_errors']} errors, {counts['parser_warnings']} warnings)",
        f"Scan mode: {_terminal(scope['mode'])}",
        f"Analysis execution: {_terminal(scope['execution'])}",
        f"Structure analysis: {_terminal(scope['structure_analysis'])}",
        f"Verdict: {verdict}",
        f"Template analysis: {_terminal(coverage.get('status', 'not_run'))}",
        f"Findings: {counts['findings']}",
        f"Breakdown: [CRITICAL: {crit_count}]  [ERROR: {err_count}]  [WARNING: {warn_count}]  [INFO: {info_count}]",
        thin_sep,
        "Container Facts:",
    ]
    for label, value in (
        ("Format version", parsed.format_version),
        ("Header length", parsed.header_length),
        ("Data start", parsed.data_start),
        ("Declared tensors", parsed.declared_tensor_count),
        ("Declared metadata", parsed.declared_metadata_count),
    ):
        if value is not None:
            lines.append(f"  • {label}: {_terminal(value)}")

    lines.extend(["", thin_sep, "Template coverage:",
                  f"  • Status: {_terminal(coverage.get('status', 'not_run'))}",
                  f"  • Loader profile: {_terminal(coverage.get('loader_profile', 'unknown'))}",
                  f"  • Inspected templates: {_terminal(coverage.get('templates_inspected', 0))}",
                  "  • Static inspection does not verify runtime exploitability or model authenticity."])
    for template in parsed.extra.get("chat_templates", []):
        lines.append(f"  • {_terminal(template.get('source'))}: {_terminal(template.get('status'))}; effective={bool(template.get('effective'))}")
    for skipped in coverage.get("skipped_sources", []):
        lines.append(f"  • Skipped {_terminal(skipped.get('source'))}: {_terminal(skipped.get('reason'))}")
    lines.extend(["", thin_sep, f"Metadata ({len(parsed.metadata)}):"])
    if parsed.metadata:
        for key, value in parsed.metadata.items():
            type_name = parsed.metadata_types.get(key)
            suffix = f" [{_terminal(type_name)}]" if type_name else ""
            val_str = str(value)
            if len(val_str) > 160:
                preview = _terminal(val_str[:140]) + f"... ({len(val_str)} chars)"
                lines.append(f"  • {_terminal(key)}{suffix}: {preview}")
            else:
                lines.append(f"  • {_terminal(key)}{suffix}: {_terminal(value)}")
    else:
        lines.append("  None")

    lines.extend(["", thin_sep, f"Tensors ({len(parsed.tensors)}):"])
    if parsed.tensors:
        for tensor in parsed.tensors:
            fields = [f"dtype={_terminal(tensor.dtype)}", f"shape={_terminal(tensor.shape)}",
                      f"offsets={_terminal(tensor.data_offsets)}"]
            if tensor.ggml_type is not None:
                fields.append(f"ggml_type={_terminal(tensor.ggml_type)}")
            if tensor.descriptor_offset is not None:
                fields.append(f"descriptor_offset={_terminal(tensor.descriptor_offset)}")
            lines.append(f"  • {_terminal(tensor.name)}: " + ", ".join(fields))
    else:
        lines.append("  None")

    if parsed.parser_errors or parsed.parser_warnings:
        lines.extend(["", thin_sep, "Parser issues:"])
        for severity, issues in (("ERROR", parsed.parser_errors), ("WARNING", parsed.parser_warnings)):
            for issue in issues:
                lines.append(f"\n  ┌── [{severity}] {_terminal(issue.code)}: {_terminal(issue.message)}")
                if issue.field is not None:
                    lines.append(f"  │   Field:    {_terminal(issue.field)}")
                if issue.evidence is not None:
                    lines.append(f"  │   Evidence: {_terminal(issue.evidence)}")
                lines.append("  └──")

    lines.extend(["", thin_sep, f"Findings ({len(result.findings)}):", thin_sep])
    if result.findings:
        for finding in result.findings:
            sev_tag = f"[{_terminal(finding.severity)}] {_terminal(finding.id)}"
            lines.append(
                f"\n  ┌── {sev_tag} ({_terminal(finding.category)}): {_terminal(finding.message)}"
            )
            if finding.field is not None:
                lines.append(f"  │   Field:       {_terminal(finding.field)}")
            if finding.evidence is not None:
                lines.append(f"  │   Evidence:    {_terminal(finding.evidence)}")
            if finding.remediation is not None:
                lines.append(f"  │   Remediation: {_terminal(finding.remediation)}")
            lines.append("  └──")
    else:
        lines.append("  No chat-template findings detected." if template_only else
                     "  No structural violations detected.")
    lines.append(sep)
    return "\n".join(lines) + "\n"


def _h(value: object) -> str:
    return html.escape(_display(value), quote=True)


def _html_optional(value: object) -> str:
    return _h(value) if value is not None else '<span class="muted">—</span>'


def _stat(label: str, value: object) -> str:
    return f'<div class="stat"><span>{html.escape(label)}</span><strong>{_h(value)}</strong></div>'


def _issue_html(issue: ParserIssue, severity: str) -> str:
    details = []
    if issue.field is not None:
        details.append(f"<dt>Field</dt><dd>{_h(issue.field)}</dd>")
    if issue.evidence is not None:
        details.append(f"<dt>Evidence</dt><dd><code>{_h(issue.evidence)}</code></dd>")
    sev_cls = html.escape(severity.lower(), quote=True)
    return (
        f'<article class="finding {sev_cls}"><div class="finding-head"><span class="badge {sev_cls}">'
        f'{html.escape(severity)}</span><strong>{_h(issue.code)}</strong></div>'
        f'<p>{_h(issue.message)}</p><dl>{"".join(details)}</dl></article>'
    )


def _finding_html(finding: Finding) -> str:
    details = [f"<dt>Category</dt><dd>{_h(finding.category)}</dd>"]
    if finding.field is not None:
        details.append(f"<dt>Field</dt><dd>{_h(finding.field)}</dd>")
    if finding.evidence is not None:
        details.append(f"<dt>Evidence</dt><dd><code>{_h(finding.evidence)}</code></dd>")
    if finding.remediation is not None:
        details.append(f"<dt>Remediation</dt><dd>{_h(finding.remediation)}</dd>")
    severity = _h(finding.severity)
    sev_cls = html.escape(finding.severity.lower(), quote=True)
    return (
        f'<article class="finding {sev_cls}"><div class="finding-head">'
        f'<span class="badge {sev_cls}">{severity}</span>'
        f'<strong>{_h(finding.id)}</strong></div><p>{_h(finding.message)}</p>'
        f'<dl>{"".join(details)}</dl></article>'
    )


def render_html(result: ScanResult) -> str:
    """Return a standalone, escaped HTML review suitable for saving as a file."""
    parsed = result.parsed
    report = report_dict(result)
    counts = report["counts"]
    assert isinstance(counts, dict)
    by_severity = counts["by_severity"]
    assert isinstance(by_severity, dict)
    status = str(report["parser_status"])
    scope = report["analysis_scope"]
    empty_findings = ("No chat-template findings detected." if scope["mode"] == "template-only" else
                      "No structural violations detected.")
    coverage = report["template_analysis"]
    template_panel = (
        '<section class="panel"><h2>Template coverage</h2>'
        f'<p><strong>Status: {_h(coverage.get("status", "not_run"))}</strong></p>'
        '<p>Static indicators do not verify runtime exploitability, malicious intent or model authenticity.</p>'
        f'<pre>{_h(coverage)}</pre></section>'
    )

    metadata_rows = "".join(
        f"<tr><th scope=\"row\">{_h(key)}</th><td>{_html_optional(parsed.metadata_types.get(key))}</td>"
        f"<td><code>{_h(value)}</code></td></tr>"
        for key, value in parsed.metadata.items()
    )
    tensor_rows = "".join(
        f"<tr><th scope=\"row\">{_h(tensor.name)}</th><td>{_html_optional(tensor.dtype)}</td>"
        f"<td><code>{_html_optional(tensor.shape)}</code></td>"
        f"<td><code>{_html_optional(tensor.data_offsets)}</code></td>"
        f"<td>{_html_optional(tensor.ggml_type)}</td>"
        f"<td>{_html_optional(tensor.descriptor_offset)}</td></tr>"
        for tensor in parsed.tensors
    )
    parser_issues = "".join(
        _issue_html(issue, severity)
        for severity, issues in (("ERROR", parsed.parser_errors), ("WARNING", parsed.parser_warnings))
        for issue in issues
    )
    findings = "".join(_finding_html(finding) for finding in result.findings)
    severity_strip = "".join(
        f'<span class="severity-count"><b>{_h(by_severity[severity])}</b> {_h(severity.title())}</span>'
        for severity in _SEVERITIES
    )
    facts = "".join(
        _stat(label, value)
        for label, value in (
            ("Format version", parsed.format_version),
            ("Header length", parsed.header_length),
            ("Data start", parsed.data_start),
            ("Declared tensors", parsed.declared_tensor_count),
            ("Declared metadata", parsed.declared_metadata_count),
        ) if value is not None
    )
    extra = (
        '<section class="panel"><h2>Additional parsed data</h2>'
        f'<pre>{_h(parsed.extra)}</pre></section>'
        if parsed.extra else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>ModelGuard review · {_h(parsed.path.name)}</title>
<style>
:root {{ color-scheme: light; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172338; background: #f5f7fb; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 42px 24px 80px; }}
header {{ padding: 30px 34px; border-radius: 20px; color: white; background: linear-gradient(125deg, #102843, #153e57 65%, #18675f); box-shadow: 0 16px 38px #10284324; }}
.eyebrow {{ margin: 0 0 11px; color: #a8e4dc; font-size: .73rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }}
h1 {{ margin: 0; font-size: clamp(1.9rem, 4vw, 2.7rem); letter-spacing: -.04em; }}
header p {{ margin: 12px 0 0; color: #d4e4ee; overflow-wrap: anywhere; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 22px 0; }}
.stat, .panel {{ background: white; border: 1px solid #dfe6ee; border-radius: 15px; box-shadow: 0 4px 16px #1a355408; }}
.stat {{ padding: 19px 21px; min-width: 0; }}
.stat span {{ display: block; color: #50647a; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
.stat strong {{ display: block; margin-top: 7px; font-size: 1.32rem; overflow-wrap: anywhere; }}
.panel {{ padding: 25px 29px; margin-top: 20px; }}
h2 {{ margin: 0 0 18px; font-size: 1.16rem; letter-spacing: -.02em; }}
.subtle, .muted {{ color: #64758a; }}
.severity-strip {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 19px; }}
.severity-count {{ background: #f1f5f9; border-radius: 999px; padding: 7px 12px; font-size: .82rem; }}
.finding-grid {{ display: grid; gap: 12px; }}
.finding {{ border: 1px solid #dfe6ee; border-left: 5px solid #6b8297; border-radius: 10px; padding: 16px 18px; background: #fff; box-shadow: 0 2px 8px #10284308; }}
.finding.critical {{ border-left-color: #dc2626; background: #fffafb; }}
.finding.error {{ border-left-color: #ea580c; background: #fffcf9; }}
.finding.warning {{ border-left-color: #d97706; background: #fffdf7; }}
.finding.info {{ border-left-color: #2563eb; background: #f8fbff; }}
.finding-head {{ display: flex; align-items: center; gap: 11px; }}
.finding p {{ margin: 12px 0 8px; line-height: 1.55; }}
.finding dl {{ display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 5px 13px; margin: 0; font-size: .88rem; }}
dt {{ color: #5a6b7f; font-weight: 700; }}
dd {{ margin: 0; overflow-wrap: anywhere; }}
.badge {{ display: inline-block; padding: 4px 9px; border-radius: 6px; background: #eef1f4; color: #364758; font-size: .67rem; font-weight: 800; letter-spacing: .05em; }}
.badge.critical {{ background: #fce8ea; color: #9f2034; }}
.badge.error {{ background: #fff0e7; color: #9b4112; }}
.badge.warning {{ background: #fff5ce; color: #745300; }}
.badge.info {{ background: #e7f3ff; color: #13548b; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .84em; overflow-wrap: anywhere; white-space: pre-wrap; }}
pre {{ background: #f6f8fb; border-radius: 9px; padding: 14px; margin: 0; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: .89rem; }}
th, td {{ border-bottom: 1px solid #e8edf2; padding: 11px 10px; vertical-align: top; overflow-wrap: anywhere; }}
thead th {{ color: #53677d; font-size: .73rem; text-transform: uppercase; letter-spacing: .05em; }}
tbody th {{ font-weight: 650; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
@media (max-width: 760px) {{ main {{ padding: 18px 12px 48px; }} header {{ padding: 25px; }} .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .panel {{ padding: 20px; }} }}
</style>
</head>
<body>
<main>
<header><p class="eyebrow">Static model inspection</p><h1>ModelGuard review</h1>
<p><strong>{_h(parsed.path.name)}</strong> · {_h(parsed.path)}</p></header>
<section class="panel"><h2>Analysis scope</h2>
<p>Scan mode: <strong>{_h(scope['mode'])}</strong> · Analysis execution: {_h(scope['execution'])}</p>
<p>Structure analysis: {_h(scope['structure_analysis'])}</p><p>{_h(scope['description'])}</p></section>
<div class="summary">
{_stat("Format", parsed.format)}{_stat("File size", f"{parsed.file_size:,} bytes")}
{_stat("Tensors", len(parsed.tensors))}{_stat("Metadata fields", len(parsed.metadata))}
{_stat("Parser status", status)}{_stat("Findings", len(result.findings))}{facts}
</div>
{template_panel}
<section class="panel"><h2>Findings</h2><div class="severity-strip">{severity_strip}</div>
<div class="finding-grid">{findings or '<p class="subtle">' + _h(empty_findings) + '</p>'}</div></section>
<section class="panel"><h2>Parser issues</h2><div class="finding-grid">{parser_issues or '<p class="subtle">No parser issues.</p>'}</div></section>
<section class="panel"><h2>Metadata</h2><div class="table-wrap"><table><thead><tr><th>Key</th><th>Type</th><th>Value</th></tr></thead>
<tbody>{metadata_rows or '<tr><td colspan="3" class="subtle">No metadata.</td></tr>'}</tbody></table></div></section>
<section class="panel"><h2>Tensors</h2><div class="table-wrap"><table><thead><tr><th>Name</th><th>Dtype</th><th>Shape</th><th>Data offsets</th><th>GGML type</th><th>Descriptor offset</th></tr></thead>
<tbody>{tensor_rows or '<tr><td colspan="6" class="subtle">No tensor descriptors.</td></tr>'}</tbody></table></div></section>
{extra}
</main>
</body>
</html>
"""

