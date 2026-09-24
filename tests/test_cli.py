"""End-to-end command-line demonstration and exports."""

import json
from pathlib import Path

from modelguard.cli import main


SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_valid_scan_writes_json_and_html(tmp_path: Path, capsys) -> None:
    json_path = tmp_path / "reports" / "valid.json"
    html_path = tmp_path / "reports" / "valid.html"
    code = main(["scan", str(SAMPLES / "valid_model.safetensors"),
                 "--json", str(json_path), "--html", str(html_path)])
    assert code == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["file"]["format"] == "SAFETENSORS"
    assert report["counts"]["tensors"] == 2
    assert report["findings"] == []
    assert "No structural violations detected" in html_path.read_text(encoding="utf-8")
    assert "No structural violations detected" in capsys.readouterr().out


def test_invalid_scan_writes_findings_and_returns_error(tmp_path: Path) -> None:
    json_path = tmp_path / "invalid.json"
    code = main(["scan", str(SAMPLES / "invalid_offset.gguf"), "--json", str(json_path)])
    assert code == 1
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert any(item["id"] == "GG118" for item in report["findings"])


def test_missing_file_is_a_controlled_failure(capsys) -> None:
    assert main(["scan", "does-not-exist.gguf"]) == 1
    assert "MG002" in capsys.readouterr().out
