"""Focused binary-format tests for the GGUF metadata reader."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modelguard import limits
from modelguard.parsers.ggml_types import GGML_TYPE_INFO
from modelguard.parsers.gguf_parser import GGUFParser


def _pack(endian: str, code: str, value: object) -> bytes:
    return struct.pack(("<" if endian == "little" else ">") + code, value)


def _string(endian: str, value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _pack(endian, "Q", len(encoded)) + encoded


def _kv(endian: str, key: str, value_type: int, value: bytes) -> bytes:
    return _string(endian, key) + _pack(endian, "I", value_type) + value


def _tensor(endian: str, name: str, shape: list[int], ggml_type: int, offset: int) -> bytes:
    return (
        _string(endian, name)
        + _pack(endian, "I", len(shape))
        + b"".join(_pack(endian, "Q", dimension) for dimension in shape)
        + _pack(endian, "I", ggml_type)
        + _pack(endian, "Q", offset)
    )


def _file(
    endian: str = "little",
    version: int = 3,
    kvs: list[bytes] | None = None,
    tensors: list[bytes] | None = None,
    payload: bytes = b"",
    alignment: int = 32,
) -> bytes:
    kvs = kvs or []
    tensors = tensors or []
    header = (
        b"GGUF"
        + _pack(endian, "I", version)
        + _pack(endian, "Q", len(tensors))
        + _pack(endian, "Q", len(kvs))
        + b"".join(kvs)
        + b"".join(tensors)
    )
    if tensors:
        header += b"\0" * (-len(header) % alignment)
    return header + payload


def _parse(tmp_path: Path, data: bytes):
    path = tmp_path / "sample.gguf"
    path.write_bytes(data)
    return GGUFParser().parse(path)


def test_v3_metadata_tensor_sizes_and_payload_is_not_read(tmp_path: Path) -> None:
    kvs = [
        _kv("little", "general.architecture", 8, _string("little", "llama")),
        _kv("little", "general.alignment", 4, _pack("little", "I", 64)),
    ]
    tensors = [
        _tensor("little", "weight", [32], 2, 0),
        _tensor("little", "bias", [2], 1, 64),
    ]
    parsed = _parse(tmp_path, _file(kvs=kvs, tensors=tensors, payload=b"\0", alignment=64))

    assert parsed.parser_errors == []
    assert parsed.format_version == 3
    assert parsed.declared_tensor_count == 2
    assert parsed.declared_metadata_count == 2
    assert parsed.metadata["general.architecture"] == "llama"
    assert parsed.metadata_types["general.alignment"] == "uint32"
    assert parsed.extra["alignment"] == 64
    assert parsed.extra["endian"] == "little"
    assert parsed.data_start == (parsed.header_length + 63) // 64 * 64
    assert [(item.name, item.dtype, item.shape, item.data_offsets) for item in parsed.tensors] == [
        ("weight", "Q4_0", [32], [0, 18]),
        ("bias", "F16", [2], [64, 68]),
    ]
    assert parsed.tensors[0].descriptor_offset is not None


def test_v2_scalar_types_arrays_and_utf8(tmp_path: Path) -> None:
    values = {
        0: ("B", 255),
        1: ("b", -4),
        2: ("H", 65535),
        3: ("h", -123),
        4: ("I", 123456),
        5: ("i", -123456),
        6: ("f", 1.5),
        7: ("B", 1),
        10: ("Q", 1 << 40),
        11: ("q", -(1 << 40)),
        12: ("d", 2.5),
    }
    kvs = [_kv("little", f"value.{type_id}", type_id, _pack("little", code, value))
           for type_id, (code, value) in values.items()]
    kvs.append(_kv("little", "value.8", 8, _string("little", "môdel")))
    array = _pack("little", "I", 3) + _pack("little", "Q", 2)
    array += _pack("little", "h", -2) + _pack("little", "h", 7)
    kvs.append(_kv("little", "value.9", 9, array))

    parsed = _parse(tmp_path, _file(version=2, kvs=kvs))

    assert parsed.parser_errors == []
    assert parsed.format_version == 2
    assert parsed.metadata["value.7"] is True
    assert parsed.metadata["value.8"] == "môdel"
    assert parsed.metadata["value.9"] == [-2, 7]
    assert parsed.metadata_types["value.9"] == "array[int16]"
    for type_id, (_, value) in values.items():
        if type_id != 7:
            assert parsed.metadata[f"value.{type_id}"] == value


def test_v3_big_endian_and_nested_arrays(tmp_path: Path) -> None:
    nested = _pack("big", "I", 9) + _pack("big", "Q", 2)
    nested += _pack("big", "I", 8) + _pack("big", "Q", 1) + _string("big", "first")
    nested += _pack("big", "I", 8) + _pack("big", "Q", 1) + _string("big", "second")
    kvs = [_kv("big", "names", 9, nested)]
    tensors = [_tensor("big", "matrix", [2, 2], 0, 0)]

    parsed = _parse(tmp_path, _file(endian="big", kvs=kvs, tensors=tensors))

    assert parsed.parser_errors == []
    assert parsed.extra["endian"] == "big"
    assert parsed.metadata["names"] == [["first"], ["second"]]
    assert parsed.tensors[0].data_offsets == [0, 16]


def test_zero_dimension_scalar_tensor_has_size(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, _file(tensors=[_tensor("little", "scalar", [], 0, 0)], payload=b"\0" * 4))
    assert parsed.parser_errors == []
    assert parsed.tensors[0].data_offsets == [0, 4]


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"BAD!" + b"\0" * 20, "MG101"),
        (b"GGUF" + struct.pack("<I", 4) + b"\0" * 16, "MG102"),
        (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0), "MG103"),
    ],
)
def test_bad_header_returns_parser_issue(tmp_path: Path, data: bytes, code: str) -> None:
    parsed = _parse(tmp_path, data)
    assert parsed.parser_errors[0].code == code


def test_unsupported_metadata_type_preserves_counts(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, _file(kvs=[_kv("little", "bad", 99, b"")]))
    assert parsed.declared_metadata_count == 1
    assert parsed.parser_errors[0].code == "MG107"


def test_unknown_tensor_type_keeps_descriptor_with_unknown_size(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, _file(tensors=[_tensor("little", "mystery", [4], 999, 0)]))
    assert parsed.parser_errors[0].code == "MG115"
    assert parsed.tensors[0].data_offsets == [0, None]
    assert parsed.tensors[0].ggml_type == 999


def test_declared_count_and_array_limits(tmp_path: Path) -> None:
    count_data = b"GGUF" + struct.pack("<IQQ", 3, limits.MAX_TENSORS + 1, 0)
    assert _parse(tmp_path, count_data).parser_errors[0].code == "MG112"
    array = _pack("little", "I", 0) + _pack("little", "Q", limits.MAX_ARRAY_ELEMENTS + 1)
    assert _parse(tmp_path, _file(kvs=[_kv("little", "large", 9, array)])).parser_errors[0].code == "MG109"


def test_invalid_utf8_and_boolean_are_rejected(tmp_path: Path) -> None:
    invalid_string = _pack("little", "Q", 1) + b"\xff"
    assert _parse(tmp_path, _file(kvs=[_kv("little", "bad", 8, invalid_string)])).parser_errors[0].code == "MG106"
    assert _parse(tmp_path, _file(kvs=[_kv("little", "bad", 7, b"\x02")])).parser_errors[0].code == "MG110"


def test_header_byte_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "MAX_HEADER_BYTES", 24)
    parsed = _parse(tmp_path, _file(kvs=[_kv("little", "name", 8, _string("little", "x"))]))
    assert parsed.parser_errors[0].code == "MG104"


def test_deprecated_ggml_ids_are_not_given_made_up_sizes() -> None:
    assert 4 not in GGML_TYPE_INFO
    assert 31 not in GGML_TYPE_INFO
    assert GGML_TYPE_INFO[42] == ("Q2_0", 64, 18)
