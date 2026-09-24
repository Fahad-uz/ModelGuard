"""Bounded static reader for GGUF v2/v3 metadata and tensor descriptors."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

from modelguard import limits
from modelguard.models.results import ParsedModel, ParserIssue, TensorDescriptor
from modelguard.parsers.base import ParserError
from modelguard.parsers.ggml_types import GGML_TYPE_INFO


_VALUE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

_VALUE_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}


class _Reader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.endian = "little"
        self.array_elements = 0

    @property
    def position(self) -> int:
        return self.stream.tell()

    def read_exact(self, size: int, field: str) -> bytes:
        if size < 0 or self.position + size > limits.MAX_HEADER_BYTES:
            raise ParserError(
                "MG104", "GGUF header exceeds the configured size limit", field,
                {"offset": self.position, "bytes": size, "limit": limits.MAX_HEADER_BYTES},
            )
        data = self.stream.read(size)
        if len(data) != size:
            raise ParserError(
                "MG103", "Truncated GGUF header", field,
                {"offset": self.position - len(data), "expected": size, "actual": len(data)},
            )
        return data

    def number(self, format_code: str, field: str) -> int | float:
        prefix = "<" if self.endian == "little" else ">"
        encoding = prefix + format_code
        return struct.unpack(encoding, self.read_exact(struct.calcsize(encoding), field))[0]

    def string(self, field: str) -> str:
        length = self.number("Q", field + ".length")
        if length > limits.MAX_STRING_BYTES:
            raise ParserError(
                "MG105", "GGUF string exceeds the configured size limit", field,
                {"length": length, "limit": limits.MAX_STRING_BYTES},
            )
        value = self.read_exact(length, field)
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParserError("MG106", "Invalid UTF-8 in GGUF string", field) from exc

    def value(self, value_type: int, field: str, depth: int = 0) -> object:
        if value_type not in _VALUE_NAMES:
            raise ParserError("MG107", "Unsupported GGUF metadata value type", field, value_type)
        if value_type == 8:
            return self.string(field)
        if value_type == 9:
            if depth >= limits.MAX_ARRAY_DEPTH:
                raise ParserError(
                    "MG108", "GGUF array nesting exceeds the configured limit", field,
                    {"limit": limits.MAX_ARRAY_DEPTH},
                )
            element_type = self.number("I", field + ".element_type")
            if element_type not in _VALUE_NAMES:
                raise ParserError("MG107", "Unsupported GGUF array element type", field, element_type)
            count = self.number("Q", field + ".length")
            if count > limits.MAX_ARRAY_ELEMENTS or self.array_elements + count > limits.MAX_ARRAY_ELEMENTS:
                raise ParserError(
                    "MG109", "GGUF array exceeds the configured element limit", field,
                    {"count": count, "limit": limits.MAX_ARRAY_ELEMENTS},
                )
            self.array_elements += count
            return [self.value(element_type, f"{field}[{index}]", depth + 1) for index in range(count)]
        value = self.number(_VALUE_FORMATS[value_type], field)
        if value_type == 7:
            if value not in (0, 1):
                raise ParserError("MG110", "Invalid GGUF boolean value", field, value)
            return bool(value)
        return value


def _value_type_name(reader: _Reader, value_type: int, field: str) -> str:
    if value_type not in _VALUE_NAMES:
        raise ParserError("MG107", "Unsupported GGUF metadata value type", field, value_type)
    if value_type == 9:
        # Array type is read with the payload; the public type string can be
        # assembled by the caller from the first type field at this position.
        position = reader.position
        subtype = reader.number("I", field + ".element_type")
        reader.stream.seek(position)
        if subtype not in _VALUE_NAMES:
            raise ParserError("MG107", "Unsupported GGUF array element type", field, subtype)
        return f"array[{_VALUE_NAMES[subtype]}]"
    return _VALUE_NAMES[value_type]


def _tensor_end(start: int, shape: list[int], ggml_type: int) -> int | None:
    info = GGML_TYPE_INFO.get(ggml_type)
    if info is None:
        return None
    _, block_size, bytes_per_block = info
    # GGUF permits zero dimensions. ggml treats missing dimensions as one.
    if (shape[0] if shape else 1) % block_size:
        return None
    elements = 1
    for dimension in shape:
        if dimension and elements > limits.MAX_TENSOR_ELEMENTS // dimension:
            return None
        elements *= dimension
    blocks = elements // block_size
    if blocks > limits.MAX_TENSOR_BYTES // bytes_per_block:
        return None
    size = blocks * bytes_per_block
    if start > limits.MAX_TENSOR_BYTES - size:
        return None
    return start + size


class GGUFParser:
    """Extract GGUF structure while leaving tensor payloads untouched."""

    def parse(self, path: Path) -> ParsedModel:
        path = Path(path)
        parsed = ParsedModel(path=path, format="GGUF", file_size=0)
        try:
            parsed.file_size = path.stat().st_size
            with path.open("rb") as stream:
                reader = _Reader(stream)
                try:
                    self._parse(reader, parsed)
                except ParserError as exc:
                    parsed.parser_errors.append(ParserIssue(exc.code, str(exc), exc.field, exc.evidence))
                    parsed.extra["parse_offset"] = reader.position
        except OSError as exc:
            parsed.parser_errors.append(ParserIssue("MG111", "Could not read GGUF file", "file", str(exc)))
        return parsed

    def _parse(self, reader: _Reader, parsed: ParsedModel) -> None:
        if reader.read_exact(4, "magic") != b"GGUF":
            raise ParserError("MG101", "Invalid GGUF magic", "magic")

        version_bytes = reader.read_exact(4, "version")
        little_version = int.from_bytes(version_bytes, "little")
        big_version = int.from_bytes(version_bytes, "big")
        if little_version in (2, 3):
            version = little_version
        elif big_version == 3:
            version = 3
            reader.endian = "big"
        else:
            raise ParserError(
                "MG102", "Unsupported GGUF version", "version",
                {"little_endian": little_version, "big_endian": big_version},
            )
        parsed.format_version = version
        parsed.extra["endian"] = reader.endian

        tensor_count = reader.number("Q", "tensor_count")
        metadata_count = reader.number("Q", "metadata_count")
        parsed.declared_tensor_count = tensor_count
        parsed.declared_metadata_count = metadata_count
        if tensor_count > limits.MAX_TENSORS:
            raise ParserError(
                "MG112", "GGUF tensor count exceeds the configured limit", "tensor_count",
                {"count": tensor_count, "limit": limits.MAX_TENSORS},
            )
        if metadata_count > limits.MAX_METADATA_FIELDS:
            raise ParserError(
                "MG113", "GGUF metadata count exceeds the configured limit", "metadata_count",
                {"count": metadata_count, "limit": limits.MAX_METADATA_FIELDS},
            )

        duplicate_keys: list[str] = []
        for index in range(metadata_count):
            field = f"metadata[{index}]"
            key = reader.string(field + ".key")
            value_type = reader.number("I", field + ".type")
            type_name = _value_type_name(reader, value_type, field)
            value = reader.value(value_type, field + ".value")
            if key in parsed.metadata:
                duplicate_keys.append(key)
            else:
                parsed.metadata[key] = value
                parsed.metadata_types[key] = type_name
        if duplicate_keys:
            parsed.extra["duplicate_metadata_keys"] = duplicate_keys

        alignment = 32
        if parsed.metadata_types.get("general.alignment") == "uint32":
            alignment = parsed.metadata["general.alignment"]
        parsed.extra["alignment"] = alignment

        for index in range(tensor_count):
            descriptor_offset = reader.position
            field = f"tensor[{index}]"
            name = reader.string(field + ".name")
            dimension_count = reader.number("I", field + ".n_dimensions")
            if dimension_count > limits.MAX_DIMENSIONS:
                raise ParserError(
                    "MG114", "GGUF tensor has too many dimensions", field + ".n_dimensions",
                    {"count": dimension_count, "limit": limits.MAX_DIMENSIONS},
                )
            shape = [reader.number("Q", f"{field}.dimensions[{i}]") for i in range(dimension_count)]
            ggml_type = reader.number("I", field + ".type")
            start = reader.number("Q", field + ".offset")
            type_info = GGML_TYPE_INFO.get(ggml_type)
            descriptor = TensorDescriptor(
                name=name,
                dtype=type_info[0] if type_info is not None else f"UNKNOWN_{ggml_type}",
                shape=shape,
                data_offsets=[start, _tensor_end(start, shape, ggml_type)],
                ggml_type=ggml_type,
                descriptor_offset=descriptor_offset,
            )
            parsed.tensors.append(descriptor)
            if type_info is None:
                parsed.parser_errors.append(
                    ParserIssue("MG115", "Unsupported GGML tensor type", field + ".type", ggml_type)
                )

        parsed.header_length = reader.position
        if alignment > 0:
            parsed.data_start = reader.position + (-reader.position % alignment) if tensor_count else reader.position


def parse_gguf(path: Path) -> ParsedModel:
    """Convenience entry point for callers that do not need a parser instance."""
    return GGUFParser().parse(path)
