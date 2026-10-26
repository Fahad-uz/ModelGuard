"""Demo fixtures exercise both the parser and analysis paths."""

from pathlib import Path

from modelguard.analyzers.template_analyzer import HAVE_JINJA2
from modelguard.scan import scan_file


SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_valid_fixtures_have_no_structural_errors() -> None:
    for filename in ("valid_model.safetensors", "valid_model.gguf"):
        report = scan_file(SAMPLES / filename)
        assert not report.parsed.parser_errors, filename
        assert not [item for item in report.findings if item.severity in ("ERROR", "CRITICAL")], filename
        assert report.parsed.tensor_count > 0


def test_invalid_fixtures_produce_controlled_findings() -> None:
    for filename in ("truncated_header.safetensors", "invalid_offsets.safetensors",
                     "truncated_header.gguf", "bad_magic.gguf", "invalid_offset.gguf"):
        report = scan_file(SAMPLES / filename)
        assert report.findings, filename
        assert any(item.severity in ("ERROR", "CRITICAL") for item in report.findings), filename


def test_malicious_chat_template_fixtures_flagged() -> None:
    for filename in ("malicious_chat_template.gguf", "malicious_exploit.gguf"):
        report = scan_file(SAMPLES / filename)
        assert not report.parsed.parser_errors, filename
        ids = {f.id for f in report.findings}
        # Must detect dunder traversal (regex), and prompt-level risks
        assert "CT101" in ids, f"{filename}: missing CT101 (dunder traversal)"
        assert "CT110" in ids, f"{filename}: missing CT110 (system role omission)"
        assert "CT111" in ids, f"{filename}: missing CT111 (backdoor phrase)"
        # AST-based checks require Jinja2
        if HAVE_JINJA2:
            assert "CT102" in ids, f"{filename}: missing CT102 (dangerous function call)"
            assert "CT104" in ids, f"{filename}: missing CT104 (sandbox escape)"
            assert "CT105" in ids, f"{filename}: missing CT105 (file inclusion)"
        # Template must have been discovered
        assert report.parsed.extra.get("chat_templates"), f"{filename}: no chat templates extracted"
