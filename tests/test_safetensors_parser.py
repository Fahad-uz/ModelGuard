"""Focused coverage for bounded, non-executing safetensors header reads."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from modelguard.limits import MAX_HEADER_BYTES
from modelguard.parsers.safetensors_parser import SafetensorsParser


def write_file(path: Path, header: bytes | dict[str, object], payload: bytes = b"") -> Path:
    if isinstance(header, dict):
        header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return path


def error_codes(path: Path) -> set[str]:
    return {issue.code for issue in SafetensorsParser().parse(path).parser_errors}


def test_parses_header_without_loading_tensors(tmp_path: Path) -> None:
    header = {
        "__metadata__": {"format": "pt", "note": "tiny fixture"},
        "weights": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "empty": {"dtype": "F16", "shape": [0, 4], "data_offsets": [8, 8]},
        "scalar": {"dtype": "I8", "shape": [], "data_offsets": [8, 9]},
    }
    path = write_file(tmp_path / "valid.safetensors", header, b"123456789")

    parsed = SafetensorsParser().parse(path)

    assert parsed.parser_errors == []
    assert parsed.path == path
    assert parsed.format == "SAFETENSORS"
    assert parsed.file_size == path.stat().st_size
    assert parsed.header_length == len(json.dumps(header, separators=(",", ":")).encode())
    assert parsed.data_start == 8 + parsed.header_length
    assert parsed.declared_tensor_count == 3
    assert parsed.declared_metadata_count == 2
    assert parsed.metadata == {"format": "pt", "note": "tiny fixture"}
    assert [(tensor.name, tensor.dtype, tensor.shape, tensor.data_offsets) for tensor in parsed.tensors] == [
        ("weights", "F32", [2], [0, 8]),
        ("empty", "F16", [0, 4], [8, 8]),
        ("scalar", "I8", [], [8, 9]),
    ]


def test_preserves_malformed_tensor_fields_for_analyser(tmp_path: Path) -> None:
    path = write_file(
        tmp_path / "malformed.safetensors",
        {
            "__metadata__": {"unexpected": 7},
            "bad": {"dtype": 12, "shape": [2, -1], "data_offsets": ["no", 10]},
            "missing": {"dtype": "F32"},
            "non_object": [1, 2],
        },
    )

    parsed = SafetensorsParser().parse(path)

    assert parsed.metadata == {"unexpected": 7}
    assert parsed.tensors[0].dtype == 12
    assert parsed.tensors[0].shape == [2, -1]
    assert parsed.tensors[0].data_offsets == ["no", 10]
    assert parsed.tensors[1].shape is None
    assert parsed.tensors[1].data_offsets is None
    assert parsed.extra["raw_tensor_entries"] == {"non_object": [1, 2]}
    assert {issue.code for issue in parsed.parser_errors} == {"STP015"}


def test_preserves_non_object_metadata(tmp_path: Path) -> None:
    path = write_file(tmp_path / "metadata.safetensors", {"__metadata__": ["bad"]})
    parsed = SafetensorsParser().parse(path)
    assert parsed.extra["raw_metadata"] == ["bad"]
    assert "STP013" in {issue.code for issue in parsed.parser_errors}


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"", "STP002"),
        (b"1234567", "STP002"),
        (struct.pack("<Q", 10) + b"{}", "STP004"),
        (struct.pack("<Q", MAX_HEADER_BYTES + 1), "STP003"),
    ],
)
def test_rejects_truncated_or_oversized_header(
    tmp_path: Path, content: bytes, expected_code: str
) -> None:
    path = tmp_path / "broken.safetensors"
    path.write_bytes(content)
    assert expected_code in error_codes(path)


@pytest.mark.parametrize(
    ("header", "expected_code"),
    [
        (b"{", "STP007"),
        (b"{\xff}", "STP005"),
        (b'{"x":1,"x":2}', "STP006"),
        (b'{"x":{"shape":[],"shape":[1]}}', "STP006"),
        (b'{"__metadata__":{"a":"1","a":"2"}}', "STP006"),
        (b"[]", "STP009"),
        (b" {}", "STP009"),
        (b"{}\t", "STP007"),
        (b'{"x":NaN}', "STP007"),
        (b'{"bad\\ud800":{}}', "STP007"),
        (b'{"__metadata__":{"bad":"\\ud800"}}', "STP007"),
    ],
)
def test_rejects_invalid_json_or_encoding(
    tmp_path: Path, header: bytes, expected_code: str
) -> None:
    path = write_file(tmp_path / "invalid.safetensors", header)
    assert expected_code in error_codes(path)


def test_allows_space_padding(tmp_path: Path) -> None:
    path = write_file(tmp_path / "padded.safetensors", b"{}   ")
    parsed = SafetensorsParser().parse(path)
    assert parsed.parser_errors == []
    assert parsed.declared_tensor_count == 0


def test_missing_file_is_controlled_error(tmp_path: Path) -> None:
    parsed = SafetensorsParser().parse(tmp_path / "missing.safetensors")
    assert parsed.file_size == 0
    assert {issue.code for issue in parsed.parser_errors} == {"STP001"}
