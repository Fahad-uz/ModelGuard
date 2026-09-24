"""Data shared by parsers, analysers, and presentation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Severity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(slots=True)
class ParserIssue:
    code: str
    message: str
    field: str | None = None
    evidence: object = None


@dataclass(slots=True)
class TensorDescriptor:
    name: str
    dtype: object = None
    shape: object = None
    data_offsets: object = None  # [start, end], relative to the data region
    ggml_type: int | None = None
    descriptor_offset: int | None = None


@dataclass(slots=True)
class ParsedModel:
    path: Path
    format: str
    file_size: int
    format_version: int | None = None
    header_length: int | None = None
    data_start: int | None = None
    declared_tensor_count: int | None = None
    declared_metadata_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    metadata_types: dict[str, str] = field(default_factory=dict)
    tensors: list[TensorDescriptor] = field(default_factory=list)
    parser_warnings: list[ParserIssue] = field(default_factory=list)
    parser_errors: list[ParserIssue] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)


@dataclass(slots=True)
class Finding:
    id: str
    severity: Severity
    category: str
    message: str
    field: str | None = None
    evidence: object = None
    remediation: str | None = None


@dataclass(slots=True)
class ScanResult:
    parsed: ParsedModel
    findings: list[Finding]
