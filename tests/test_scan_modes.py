"""Concurrent orchestration, independent failures and template-only scope."""

import json
from pathlib import Path
from threading import Barrier, get_ident

import pytest

from modelguard.analyzers.template_analyzer import TemplateAnalyzer
from modelguard.cli import main
from modelguard.models.results import Finding, ParserIssue
from modelguard.review_output import render_html, render_terminal, report_dict
from modelguard.scan import StructureAnalyzer, scan_file


SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_default_analyses_overlap_parse_once_and_merge_in_stable_order(monkeypatch):
    from modelguard.parsers.gguf_parser import GGUFParser

    rendezvous = Barrier(2, timeout=3)
    threads = {}
    parses = []
    original_parse = GGUFParser.parse

    def parse(self, path):
        parses.append(path)
        return original_parse(self, path)

    def structural(self, parsed):
        threads["structure"] = get_ident()
        rendezvous.wait()
        return [Finding("STRUCT", "INFO", "test", "Structural result")]

    def templates(self, parsed, *, profile):
        threads["template"] = get_ident()
        rendezvous.wait()
        parsed.extra["template_analysis"] = {"status": "completed", "complete": True}
        return [Finding("TEMPLATE", "INFO", "test", "Template result")]

    monkeypatch.setattr(GGUFParser, "parse", parse)
    monkeypatch.setattr(StructureAnalyzer, "analyze", structural)
    monkeypatch.setattr(TemplateAnalyzer, "analyze", templates)
    result = scan_file(SAMPLES / "valid_model.gguf")
    assert len(parses) == 1
    assert len(set(threads.values())) == 2
    assert [f.id for f in result.findings] == ["STRUCT", "TEMPLATE"]
    assert result.parsed.extra["analysis_scope"] == {
        "mode": "full", "execution": "concurrent", "structure_analysis": "completed"}


@pytest.mark.parametrize("failed_stage", ["structure", "template"])
def test_one_failed_analyzer_keeps_the_other_result(monkeypatch, failed_stage):
    rendezvous = Barrier(2, timeout=3)

    def structural(self, parsed):
        rendezvous.wait()
        if failed_stage == "structure":
            raise RuntimeError("structure failed")
        return [Finding("STRUCT", "WARNING", "test", "Structural result")]

    def templates(self, parsed, *, profile):
        rendezvous.wait()
        if failed_stage == "template":
            raise RuntimeError("template failed")
        parsed.extra["template_analysis"] = {"status": "completed", "complete": True}
        return [Finding("TEMPLATE", "WARNING", "test", "Template result")]

    monkeypatch.setattr(StructureAnalyzer, "analyze", structural)
    monkeypatch.setattr(TemplateAnalyzer, "analyze", templates)
    result = scan_file(SAMPLES / "valid_model.gguf")
    ids = [f.id for f in result.findings]
    assert ids == (["MG998", "TEMPLATE"] if failed_stage == "structure" else ["STRUCT", "CT999"])
    if failed_stage == "structure":
        assert result.parsed.extra["analysis_scope"]["structure_analysis"] == "failed"
        assert result.parsed.extra["template_analysis"]["complete"]
    else:
        assert not result.parsed.extra["template_analysis"]["complete"]


def test_template_only_skips_structural_checks_and_labels_all_reports(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Structural analysis ran in template-only mode")

    monkeypatch.setattr(StructureAnalyzer, "analyze", forbidden)
    # Offset errors are structural findings, not container parser errors.
    result = scan_file(SAMPLES / "invalid_offset.gguf", template_only=True)
    assert not result.findings
    report = report_dict(result)
    assert report["analysis_scope"]["mode"] == "template-only"
    assert report["analysis_scope"]["structure_analysis"] == "not_requested"
    assert report["template_analysis"]["status"] == "absent"
    for output in (render_terminal(result), render_html(result)):
        assert "template-only" in output
        assert "not_requested" in output
        assert "No chat-template findings detected." in output
        assert "No structural violations detected" not in output


def test_template_only_retains_all_template_findings():
    path = SAMPLES / "malicious_chat_template.gguf"
    full = scan_file(path)
    template_only = scan_file(path, template_only=True)
    assert [f.id for f in template_only.findings] == [f.id for f in full.findings if f.id.startswith("CT")]
    assert {"CT101", "CT102", "CT110", "CT111"}.issubset({f.id for f in template_only.findings})
    assert template_only.parsed.extra["template_analysis"]["complete"]


@pytest.mark.parametrize("filename", ["bad_magic.gguf", "truncated_header.gguf", "truncated_header.safetensors"])
def test_template_only_preserves_parser_errors(filename):
    result = scan_file(SAMPLES / filename, template_only=True)
    errors = result.parsed.parser_errors
    assert errors
    assert {i.code for i in errors}.issubset({f.id for f in result.findings if f.severity == "ERROR"})
    assert not result.parsed.extra["template_analysis"]["complete"]


def test_template_only_preserves_parser_warnings(monkeypatch):
    from modelguard.parsers.gguf_parser import GGUFParser

    original_parse = GGUFParser.parse

    def parse(self, path):
        parsed = original_parse(self, path)
        parsed.parser_warnings.append(ParserIssue("TEST_WARNING", "Parser warning"))
        return parsed

    monkeypatch.setattr(GGUFParser, "parse", parse)
    result = scan_file(SAMPLES / "valid_model.gguf", template_only=True)
    assert any(f.id == "TEST_WARNING" and f.severity == "WARNING" for f in result.findings)


@pytest.mark.parametrize("template_only", [False, True])
def test_missing_file_runs_neither_analyzer(monkeypatch, tmp_path, template_only):
    def forbidden(*args, **kwargs):
        pytest.fail("Analyzer ran without an identifiable container")

    monkeypatch.setattr(StructureAnalyzer, "analyze", forbidden)
    monkeypatch.setattr(TemplateAnalyzer, "analyze", forbidden)
    result = scan_file(tmp_path / "missing.gguf", template_only=template_only)
    assert [f.id for f in result.findings] == ["MG002"]
    assert result.parsed.extra["analysis_scope"]["execution"] == "not_run"
    assert result.parsed.extra["template_analysis"]["status"] == "not_run"


@pytest.mark.parametrize("flag", ["--template-only", "--chat-template-only"])
def test_cli_template_only_exports_and_profile(tmp_path, flag, capsys):
    json_path, html_path = tmp_path / "report.json", tmp_path / "report.html"
    assert main(["scan", str(SAMPLES / "invalid_offset.gguf"), flag,
                 "--template-profile", "processor", "--json", str(json_path), "--html", str(html_path)]) == 0
    report = json.loads(json_path.read_text())
    assert report["analysis_scope"]["structure_analysis"] == "not_requested"
    assert report["template_analysis"]["loader_profile"] == "processor"
    assert "template-only" in html_path.read_text()
    assert "template-only" in capsys.readouterr().out


@pytest.mark.parametrize("filename", ["malicious_chat_template.gguf", "truncated_header.gguf", "missing.gguf"])
def test_cli_template_only_failure_exit_codes(filename):
    assert main(["scan", str(SAMPLES / filename), "--template-only"]) == 1
