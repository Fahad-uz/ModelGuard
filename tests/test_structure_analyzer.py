"""Behaviour of structural checks on normalized parser facts."""

from pathlib import Path

from modelguard.analyzers.structure_analyzer import StructureAnalyzer
from modelguard.models.results import ParsedModel, ParserIssue, TensorDescriptor


def codes(parsed: ParsedModel) -> set[str]:
    return {finding.id for finding in StructureAnalyzer().analyze(parsed)}


def test_valid_safetensors_layout_has_no_violations() -> None:
    model = ParsedModel(Path("x.safetensors"), "SAFETENSORS", file_size=82, data_start=74,
                        tensors=[TensorDescriptor("w", "F32", [2], [0, 8])])
    assert codes(model) == set()


def test_safetensors_reports_shape_size_overlap_and_bounds() -> None:
    model = ParsedModel(Path("x.safetensors"), "SAFETENSORS", file_size=90, data_start=82,
                        tensors=[TensorDescriptor("a", "F32", [2], [0, 8]),
                                 TensorDescriptor("b", "F32", [2], [4, 13]),
                                 TensorDescriptor("c", "F32", [-1], [13, 14])])
    assert {"ST106", "ST111", "ST114", "ST113"} <= codes(model)


def test_safetensors_reports_missing_fields_and_unindexed_bytes() -> None:
    model = ParsedModel(Path("x.safetensors"), "SAFETENSORS", file_size=84, data_start=80,
                        tensors=[TensorDescriptor("a", None, None, None)])
    assert {"ST103", "ST105", "ST109", "ST116"} <= codes(model)


def test_parser_error_becomes_a_finding() -> None:
    model = ParsedModel(Path("x.gguf"), "GGUF", 3,
                        parser_errors=[ParserIssue("MG100", "Truncated GGUF header", "header")])
    assert "MG100" in codes(model)


def test_gguf_reports_truncated_tensor_region_and_bad_alignment() -> None:
    model = ParsedModel(Path("x.gguf"), "GGUF", file_size=132, format_version=3, data_start=128,
                        declared_tensor_count=1, tensors=[TensorDescriptor("w", "F32", [2], [4, 12], ggml_type=0)],
                        extra={"alignment": 32})
    assert {"GG113", "GG118"} <= codes(model)


def test_gguf_quantized_first_dimension_must_match_block_size() -> None:
    model = ParsedModel(Path("x.gguf"), "GGUF", file_size=200, format_version=3, data_start=128,
                        declared_tensor_count=1, tensors=[TensorDescriptor("w", "Q4_0", [31], [0, 18], ggml_type=2)],
                        extra={"alignment": 32})
    assert "GG115" in codes(model)


def test_gguf_reports_duplicate_metadata_and_wrong_alignment_type() -> None:
    model = ParsedModel(Path("x.gguf"), "GGUF", file_size=128, format_version=3, data_start=128,
                        declared_metadata_count=2, metadata={"general.alignment": "32"},
                        metadata_types={"general.alignment": "string"},
                        extra={"alignment": 32, "duplicate_metadata_keys": ["general.alignment"]})
    assert {"GG102", "GG123", "GG124"} <= codes(model)


def test_gguf_descriptor_order_must_follow_packed_offsets() -> None:
    model = ParsedModel(Path("x.gguf"), "GGUF", file_size=168, format_version=3, data_start=128,
                        declared_tensor_count=2,
                        tensors=[TensorDescriptor("second", "F32", [2], [32, 40], ggml_type=0),
                                 TensorDescriptor("first", "F32", [2], [0, 8], ggml_type=0)],
                        extra={"alignment": 32})
    assert "GG120" in codes(model)
