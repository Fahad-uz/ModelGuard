"""Common interface and format detection for static model parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from modelguard.models.results import ParsedModel


class ParserError(Exception):
    """A controlled failure while reading an untrusted model container."""

    def __init__(self, code: str, message: str, field: str | None = None, evidence: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.evidence = evidence


class ModelParser(Protocol):
    def parse(self, path: Path) -> ParsedModel:
        """Extract container facts without loading tensors or executing content."""


def detect_format(path: Path) -> str:
    """Use magic where possible, then extension for damaged headers."""
    with path.open("rb") as stream:
        prefix = stream.read(16)
    if prefix.startswith(b"GGUF"):
        return "GGUF"
    if path.suffix.lower() == ".gguf":
        return "GGUF"
    if path.suffix.lower() == ".safetensors":
        return "SAFETENSORS"
    if len(prefix) >= 9 and prefix[8:9] == b"{":
        return "SAFETENSORS"
    raise ParserError("MG001", "Could not identify GGUF or safetensors format", "format")
