"""Presentation of normalized results, including hostile model-supplied text."""

import json
from pathlib import Path

from modelguard.models.results import Finding, ParsedModel, ParserIssue, ScanResult, TensorDescriptor
from modelguard.review_output import render_html, render_terminal, report_dict


def sample_result() -> ScanResult:
    parsed = ParsedModel(
        path=Path("models/<script>alert(1).gguf"),
        format="GGUF",
        file_size=256,
        format_version=3,
        data_start=128,
        declared_tensor_count=1,
        declared_metadata_count=1,
        metadata={"<img src=x onerror=alert(1)>": 'a & "quoted" value'},
        metadata_types={"<img src=x onerror=alert(1)>": "STRING"},
        tensors=[TensorDescriptor("<b>weight</b>", "F32", [2], [0, 8], ggml_type=0)],
        parser_warnings=[ParserIssue("P001", "Suspicious <tag>", "<field>", {"raw": "<&>"})],
        extra={"alignment": 32},
    )
    return ScanResult(
        parsed=parsed,
        findings=[Finding("GG101", "ERROR", "version", "Bad <script> value", "version",
                          {"value": "<img src=x>"}, "Check & repair")],
    )


def test_report_dict_is_json_ready_and_includes_scan_details() -> None:
    report = report_dict(sample_result())

    assert report["file"]["name"] == "<script>alert(1).gguf"
    assert report["parser_status"] == "WARNING"
    assert report["counts"]["by_severity"]["ERROR"] == 1
    assert report["counts"]["tensors"] == 1
    assert report["parsed"]["format_version"] == 3
    assert report["parsed"]["metadata"]["<img src=x onerror=alert(1)>"] == 'a & "quoted" value'
    assert report["parsed"]["tensors"][0]["ggml_type"] == 0
    assert report["parsed"]["parser_warnings"][0]["code"] == "P001"
    assert report["findings"][0]["evidence"] == {"value": "<img src=x>"}
    json.dumps(report, allow_nan=False)


def test_html_is_standalone_and_escapes_untrusted_content() -> None:
    output = render_html(sample_result())

    assert output.startswith("<!doctype html>")
    assert "<style>" in output and "</html>" in output
    assert "Metadata" in output and "Tensors" in output and "Findings" in output
    assert "Data offsets" in output and "Parser issues" in output
    assert "&lt;script&gt;alert(1).gguf" in output
    assert "&lt;img src=x onerror=alert(1)&gt;" in output
    assert "&lt;b&gt;weight&lt;/b&gt;" in output
    assert "&lt;tag&gt;" in output and "&lt;field&gt;" in output
    assert "a &amp; &quot;quoted&quot; value" in output
    assert "Bad &lt;script&gt; value" in output
    assert "Check &amp; repair" in output
    assert "<script>" not in output
    assert "<img src=x" not in output


def test_terminal_shows_evidence_and_escapes_control_characters() -> None:
    result = sample_result()
    result.findings[0].message = "Unexpected\x1b[31m value\nnext line"

    output = render_terminal(result)

    assert "Parser: WARNING" in output
    assert "File: models/<script>alert(1).gguf" in output
    assert "Metadata (1):" in output and "Tensors (1):" in output
    assert "[ERROR] GG101" in output
    assert "Evidence:" in output and "Remediation:" in output
    assert "\\u001b[31m" in output and "\\nnext line" in output
    assert "\x1b" not in output


def test_parser_error_takes_priority_over_warning() -> None:
    result = sample_result()
    result.parsed.parser_errors.append(ParserIssue("P002", "Cannot finish"))

    assert report_dict(result)["parser_status"] == "ERROR"
    assert "Parser: ERROR (1 errors, 1 warnings)" in render_terminal(result)
