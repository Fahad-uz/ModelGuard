"""Create tiny, deterministic GGUF and safetensors review fixtures."""

from __future__ import annotations

import json
import struct
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent


def safetensors_file(header: dict[str, object], data: bytes) -> bytes:
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + data


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def gguf_file(*, tensor_offset: int = 0, alignment: int = 32) -> bytes:
    # GGUF v3, little-endian; one F32 tensor with two elements.
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 2))
    header += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("demo")
    header += gguf_string("general.alignment") + struct.pack("<II", 4, alignment)
    header += gguf_string("weight") + struct.pack("<I", 1) + struct.pack("<Q", 2)
    header += struct.pack("<IQ", 0, tensor_offset)
    header += b"\x00" * (-len(header) % 32)
    header += struct.pack("<2f", 1.0, -1.0)
    return bytes(header)


def main() -> None:
    valid_header = {
        "__metadata__": {"source": "synthetic", "purpose": "first review"},
        "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "bias": {"dtype": "F16", "shape": [2], "data_offsets": [8, 12]},
    }
    valid_data = struct.pack("<2f2e", 1.0, -1.0, 0.5, -0.5)
    (DESTINATION / "valid_model.safetensors").write_bytes(safetensors_file(valid_header, valid_data))
    (DESTINATION / "truncated_header.safetensors").write_bytes(struct.pack("<Q", 500) + b"{\"x\":")
    invalid_header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 1000]}}
    (DESTINATION / "invalid_offsets.safetensors").write_bytes(safetensors_file(invalid_header, struct.pack("<2f", 1.0, 2.0)))
    (DESTINATION / "valid_model.gguf").write_bytes(gguf_file())
    (DESTINATION / "truncated_header.gguf").write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x01")
    (DESTINATION / "bad_magic.gguf").write_bytes(b"BAD!" + gguf_file()[4:])
    (DESTINATION / "invalid_offset.gguf").write_bytes(gguf_file(tensor_offset=32))


if __name__ == "__main__":
    main()
