"""Read safetensors headers without touching tensor payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import BinaryIO

from modelguard.limits import MAX_HEADER_BYTES, MAX_METADATA_FIELDS, MAX_STRING_BYTES, MAX_TENSORS
from modelguard.models.results import ParsedModel, ParserIssue, TensorDescriptor
from modelguard.parsers.base import ParserError


_HEADER_LENGTH_BYTES = 8


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Keep JSON's original pairs long enough to reject duplicate names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ParserError("STP006", f"Duplicate safetensors JSON key: {key!r}", key)
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    # Python's JSON decoder otherwise accepts NaN and Infinity, which are not JSON.
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _check_json_strings(root: object) -> None:
    """Reject lone surrogate escapes, which cannot represent UTF-8 strings."""
    pending = [root]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ParserError("STP007", "Invalid Unicode escape in safetensors header", "header") from error
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


class SafetensorsParser:
    """Extract the bounded JSON header and leave structural checks to the analyser."""

    def parse(self, path: Path) -> ParsedModel:
        path = Path(path)
        parsed = ParsedModel(path=path, format="SAFETENSORS", file_size=0)
        try:
            with path.open("rb") as stream:
                parsed.file_size = os.fstat(stream.fileno()).st_size
                header = self._read_header(stream, parsed)
                self._extract_header(header, parsed)
        except ParserError as error:
            parsed.parser_errors.append(
                ParserIssue(error.code, str(error), error.field, error.evidence)
            )
        except OSError as error:
            parsed.parser_errors.append(
                ParserIssue("STP001", f"Cannot read safetensors file: {error}", "file")
            )
        return parsed

    @staticmethod
    def _read_header(stream: BinaryIO, parsed: ParsedModel) -> dict[str, object]:
        length_bytes = stream.read(_HEADER_LENGTH_BYTES)
        if len(length_bytes) != _HEADER_LENGTH_BYTES:
            raise ParserError(
                "STP002", "File is shorter than the 8-byte safetensors header length", "header_length"
            )

        header_length = int.from_bytes(length_bytes, "little", signed=False)
        parsed.header_length = header_length
        parsed.data_start = _HEADER_LENGTH_BYTES + header_length
        if header_length > MAX_HEADER_BYTES:
            raise ParserError(
                "STP003",
                f"Safetensors header exceeds the {MAX_HEADER_BYTES}-byte limit",
                "header_length",
                header_length,
            )
        if parsed.data_start > parsed.file_size:
            raise ParserError(
                "STP004", "Safetensors header extends past the end of the file", "header_length",
                {"header_length": header_length, "file_size": parsed.file_size},
            )

        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ParserError(
                "STP004", "Safetensors header ended before its declared length", "header_length",
                {"header_length": header_length, "bytes_read": len(raw_header)},
            )
        try:
            header_text = raw_header.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ParserError("STP005", f"Safetensors header is not UTF-8: {error}", "header") from error
        if not raw_header.startswith(b"{"):
            raise ParserError("STP009", "Safetensors header must begin with '{'", "header")

        # Check the unconsumed suffix because only 0x20 padding is permitted.
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        try:
            header, end = decoder.raw_decode(header_text)
        except ParserError:
            raise
        except (json.JSONDecodeError, ValueError, RecursionError) as error:
            raise ParserError("STP007", f"Invalid safetensors JSON header: {error}", "header") from error
        if any(char != " " for char in header_text[end:]):
            raise ParserError("STP007", "Invalid data after safetensors JSON header", "header")
        if not isinstance(header, dict):
            raise ParserError("STP008", "Safetensors header must be a JSON object", "header")
        _check_json_strings(header)
        return header

    @staticmethod
    def _extract_header(header: dict[str, object], parsed: ParsedModel) -> None:
        tensor_count = len(header) - ("__metadata__" in header)
        parsed.declared_tensor_count = tensor_count
        if tensor_count > MAX_TENSORS:
            raise ParserError(
                "STP010", f"Safetensors header declares more than {MAX_TENSORS} tensors",
                "tensors", tensor_count,
            )

        if "__metadata__" in header:
            metadata = header["__metadata__"]
            if isinstance(metadata, dict):
                parsed.declared_metadata_count = len(metadata)
                if len(metadata) > MAX_METADATA_FIELDS:
                    raise ParserError(
                        "STP011", f"Safetensors metadata exceeds {MAX_METADATA_FIELDS} fields",
                        "__metadata__", len(metadata),
                    )
                parsed.metadata = metadata
                for key, value in metadata.items():
                    parsed.metadata_types[key] = type(value).__name__
                    if len(key.encode("utf-8")) > MAX_STRING_BYTES or (
                        isinstance(value, str) and len(value.encode("utf-8")) > MAX_STRING_BYTES
                    ):
                        parsed.parser_errors.append(
                            ParserIssue(
                                "STP012", "Safetensors metadata string exceeds the size limit",
                                f"__metadata__.{key}",
                            )
                        )
            else:
                parsed.extra["raw_metadata"] = metadata
                parsed.parser_errors.append(
                    ParserIssue("STP013", "Safetensors __metadata__ must be an object", "__metadata__", metadata)
                )

        raw_tensor_entries: dict[str, object] = {}
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if len(name.encode("utf-8")) > MAX_STRING_BYTES:
                parsed.parser_errors.append(
                    ParserIssue("STP014", "Safetensors tensor name exceeds the size limit", name)
                )
            if isinstance(value, dict):
                parsed.tensors.append(
                    TensorDescriptor(
                        name=name,
                        dtype=value.get("dtype"),
                        shape=value.get("shape"),
                        data_offsets=value.get("data_offsets"),
                    )
                )
            else:
                raw_tensor_entries[name] = value
                parsed.tensors.append(TensorDescriptor(name=name))
                parsed.parser_errors.append(
                    ParserIssue("STP015", "Safetensors tensor descriptor must be an object", name, value)
                )
        if raw_tensor_entries:
            parsed.extra["raw_tensor_entries"] = raw_tensor_entries
