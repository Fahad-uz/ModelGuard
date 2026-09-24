"""Demo fixtures exercise both the parser and analysis paths."""

from pathlib import Path

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
