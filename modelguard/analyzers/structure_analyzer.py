"""Evaluate container facts without touching model tensor bytes."""

from __future__ import annotations

from collections import Counter

from modelguard.limits import MAX_TENSOR_BYTES, MAX_TENSOR_ELEMENTS, SUSPICIOUS_DIMENSION
from modelguard.models.results import Finding, ParsedModel, Severity


# Widths come from safetensors' Dtype::bits(), including sub-byte formats.
SAFETENSORS_DTYPE_BITS: dict[str, int] = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8,
    "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64, "C64": 64,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}


def _finding(code: str, severity: Severity, category: str, message: str,
             field: str | None = None, evidence: object = None,
             remediation: str | None = None) -> Finding:
    return Finding(code, severity, category, message, field, evidence, remediation)


def _checked_elements(shape: object, *, gguf: bool = False) -> tuple[int | None, list[Finding]]:
    prefix = "GG" if gguf else "ST"
    issues: list[Finding] = []
    if not isinstance(shape, list) or any(type(d) is not int for d in shape):
        return None, [_finding(prefix + "105", "ERROR", "shape", "Shape must be a list of integer dimensions", evidence=shape)]
    if gguf and len(shape) > 4:
        issues.append(_finding("GG105", "ERROR", "shape", "GGUF tensor has more than four dimensions", evidence=shape))
    product = 1
    for dim in shape:
        if dim < 0:
            issues.append(_finding(prefix + "106", "ERROR", "shape", "Negative tensor dimension", evidence=dim))
            return None, issues
        if dim > SUSPICIOUS_DIMENSION:
            issues.append(_finding(prefix + "107", "WARNING", "shape", "Unusually large tensor dimension", evidence=dim))
        if dim and product > MAX_TENSOR_ELEMENTS // dim:
            issues.append(_finding(prefix + "108", "ERROR", "resource_limit", "Tensor element count exceeds safe 64-bit limit", evidence=shape))
            return None, issues
        product *= dim
    return product, issues


def _offsets(value: object) -> tuple[int, int] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v < 0 for v in value)):
        return None
    return value[0], value[1]


class StructureAnalyzer:
    """Turn parsed facts and controlled parser errors into specific findings."""

    def analyze(self, model: ParsedModel) -> list[Finding]:
        findings = [
            _finding(issue.code, "ERROR", "parser", issue.message, issue.field, issue.evidence)
            for issue in model.parser_errors
        ]
        findings.extend(
            _finding(issue.code, "WARNING", "parser", issue.message, issue.field, issue.evidence)
            for issue in model.parser_warnings
        )
        if model.format == "SAFETENSORS":
            findings.extend(self._safetensors(model))
        elif model.format == "GGUF":
            findings.extend(self._gguf(model))
        return findings

    def _safetensors(self, model: ParsedModel) -> list[Finding]:
        findings: list[Finding] = []
        if model.header_length is not None and model.header_length + 8 > model.file_size:
            findings.append(_finding("ST101", "ERROR", "header", "Header extends beyond the file", "header_length", model.header_length))
        metadata = model.metadata
        if not isinstance(metadata, dict) or any(type(k) is not str or type(v) is not str for k, v in metadata.items()):
            findings.append(_finding("ST102", "ERROR", "metadata", "__metadata__ must map strings to strings", "__metadata__"))
        if model.data_start is None:
            return findings
        available = max(0, model.file_size - model.data_start)
        regions: list[tuple[int, int, str]] = []
        for tensor in model.tensors:
            field = f"tensor.{tensor.name}"
            if not isinstance(tensor.dtype, str) or tensor.dtype not in SAFETENSORS_DTYPE_BITS:
                findings.append(_finding("ST103", "ERROR", "dtype", "Unsupported or malformed dtype", field, tensor.dtype))
                bits = None
            else:
                bits = SAFETENSORS_DTYPE_BITS[tensor.dtype]
            count, shape_issues = _checked_elements(tensor.shape)
            for issue in shape_issues:
                issue.field = field + ".shape"
            findings.extend(shape_issues)
            offsets = _offsets(tensor.data_offsets)
            if offsets is None:
                findings.append(_finding("ST109", "ERROR", "offset", "data_offsets must contain two non-negative integers", field, tensor.data_offsets))
                continue
            start, end = offsets
            if start > end:
                findings.append(_finding("ST110", "ERROR", "offset", "Tensor start offset exceeds end offset", field, tensor.data_offsets))
                continue
            if end > available:
                findings.append(_finding("ST111", "ERROR", "bounds", "Tensor data extends beyond the file", field, {"end": end, "available": available}))
            if bits is not None and count is not None:
                bit_count = count * bits
                if bit_count > MAX_TENSOR_BYTES * 8:
                    findings.append(_finding("ST112", "ERROR", "resource_limit", "Declared tensor byte size exceeds safe limit", field, bit_count))
                elif bit_count % 8 or (end - start) != bit_count // 8:
                    findings.append(_finding("ST113", "ERROR", "size", "Tensor offsets disagree with dtype and shape byte size", field,
                                             {"declared_bytes": end - start, "expected_bits": bit_count}))
            regions.append((start, end, tensor.name))
        cursor = 0
        for start, end, name in sorted(regions):
            if start < cursor:
                findings.append(_finding("ST114", "ERROR", "layout", "Tensor data regions overlap", f"tensor.{name}", {"start": start, "previous_end": cursor}))
            elif start > cursor:
                findings.append(_finding("ST115", "ERROR", "layout", "Unindexed gap in tensor data", f"tensor.{name}", {"gap_start": cursor, "gap_end": start}))
            cursor = max(cursor, end)
        if cursor < available:
            findings.append(_finding("ST116", "ERROR", "layout", "Trailing tensor data is not indexed by the header", "data_region", {"indexed_end": cursor, "available": available}))
        return findings

    def _gguf(self, model: ParsedModel) -> list[Finding]:
        # Type traits are kept with the GGUF parser and based on ggml's type table.
        from modelguard.parsers.ggml_types import GGML_TYPE_INFO

        findings: list[Finding] = []
        if model.format_version is not None and model.format_version not in (2, 3):
            findings.append(_finding("GG101", "ERROR", "version", "Unsupported GGUF version", "version", model.format_version))
        if model.header_length is not None and model.header_length > model.file_size:
            findings.append(_finding("GG121", "ERROR", "header", "GGUF descriptors extend beyond the file", "header_length", model.header_length))
        if model.declared_metadata_count is not None and model.declared_metadata_count != len(model.metadata):
            findings.append(_finding("GG102", "ERROR", "metadata", "Metadata count differs from parsed keys", "metadata_count",
                                     {"declared": model.declared_metadata_count, "parsed": len(model.metadata)}))
        if model.declared_tensor_count is not None and model.declared_tensor_count != len(model.tensors):
            findings.append(_finding("GG103", "ERROR", "tensor", "Tensor count differs from parsed descriptors", "tensor_count",
                                     {"declared": model.declared_tensor_count, "parsed": len(model.tensors)}))
        names = Counter(t.name for t in model.tensors)
        for name, count in names.items():
            if count > 1:
                findings.append(_finding("GG104", "ERROR", "tensor", "Duplicate tensor name", f"tensor.{name}", count))
            if not name or len(name.encode("utf-8")) >= 64:
                findings.append(_finding("GG122", "ERROR", "tensor", "Tensor name is empty or exceeds GGML's 63-byte limit", f"tensor.{name}", len(name.encode("utf-8"))))
        for key in model.extra.get("duplicate_metadata_keys", []):
            findings.append(_finding("GG123", "ERROR", "metadata", "Duplicate metadata key", str(key)))
        if "general.alignment" in model.metadata and model.metadata_types.get("general.alignment") != "uint32":
            findings.append(_finding("GG124", "ERROR", "alignment", "general.alignment must have GGUF uint32 type", "general.alignment",
                                     model.metadata_types.get("general.alignment")))
        alignment = model.extra.get("alignment", 32)
        if type(alignment) is not int or alignment < 8 or alignment & (alignment - 1):
            findings.append(_finding("GG109", "ERROR", "alignment", "general.alignment must be a power of two of at least 8", "general.alignment", alignment))
            alignment = 32
        if model.data_start is None:
            return findings
        if model.data_start > model.file_size:
            findings.append(_finding("GG125", "ERROR", "bounds", "Tensor data region begins beyond the file", "data_start", model.data_start))
        if model.tensors and model.data_start % alignment:
            findings.append(_finding("GG110", "ERROR", "alignment", "Tensor data region is not aligned", "data_start", model.data_start))
        available = max(0, model.file_size - model.data_start)
        regions: list[tuple[int, int, str]] = []
        for tensor in model.tensors:
            field = f"tensor.{tensor.name}"
            count, shape_issues = _checked_elements(tensor.shape, gguf=True)
            for issue in shape_issues:
                issue.field = field + ".shape"
            findings.extend(shape_issues)
            if tensor.ggml_type not in GGML_TYPE_INFO:
                findings.append(_finding("GG111", "WARNING", "dtype", "GGML tensor type has no supported size rule", field, tensor.ggml_type))
                continue
            type_name, block_size, block_bytes = GGML_TYPE_INFO[tensor.ggml_type]
            offsets = _offsets(tensor.data_offsets)
            if offsets is None:
                findings.append(_finding("GG112", "ERROR", "offset", "Invalid tensor offset or unavailable end offset", field, tensor.data_offsets))
                continue
            start, end = offsets
            if start % alignment:
                findings.append(_finding("GG113", "ERROR", "alignment", "Tensor offset is not aligned", field, {"offset": start, "alignment": alignment}))
            if start > end:
                findings.append(_finding("GG114", "ERROR", "offset", "Tensor end precedes start", field, tensor.data_offsets))
                continue
            if isinstance(tensor.shape, list) and tensor.shape and all(type(d) is int for d in tensor.shape) and tensor.shape[0] % block_size:
                findings.append(_finding("GG115", "ERROR", "shape", f"First dimension must be a multiple of {block_size} for {type_name}", field, tensor.shape))
            if count is not None and count % block_size == 0:
                expected = count // block_size * block_bytes
                if expected > MAX_TENSOR_BYTES:
                    findings.append(_finding("GG116", "ERROR", "resource_limit", "Declared tensor byte size exceeds safe limit", field, expected))
                elif end - start != expected:
                    findings.append(_finding("GG117", "ERROR", "size", "Tensor range disagrees with GGML type and shape", field,
                                             {"range_bytes": end - start, "expected_bytes": expected}))
            if end > available:
                findings.append(_finding("GG118", "ERROR", "bounds", "Tensor data extends beyond the file", field, {"end": end, "available": available}))
            regions.append((start, end, tensor.name))
        cursor = 0
        for start, end, name in regions:
            if start < cursor:
                findings.append(_finding("GG119", "ERROR", "layout", "Tensor data regions overlap", f"tensor.{name}", {"start": start, "previous_end": cursor}))
            elif start != cursor:
                findings.append(_finding("GG120", "WARNING", "layout", "Tensor offset differs from the packed GGUF layout", f"tensor.{name}",
                                         {"offset": start, "expected": cursor}))
            cursor = max(cursor, (end + alignment - 1) // alignment * alignment)
        return findings
