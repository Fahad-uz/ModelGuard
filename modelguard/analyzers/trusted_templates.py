"""Format resemblance examples and separately identified upstream snapshots."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from modelguard import limits


@dataclass(slots=True)
class CanonicalReference:
    name: str
    description: str
    supported_architectures: tuple[str, ...]
    canonical_source: str
    key_delimiters: tuple[str, ...]
    expected_roles: tuple[str, ...] = ("system", "user", "assistant")


# Historical API name retained for callers. These hand-written FORMAT EXAMPLES
# have no upstream identity and are never used as evidence of authenticity.
CANONICAL_REFERENCES: dict[str, CanonicalReference] = {
    "chatml": CanonicalReference(
        name="ChatML",
        description="Standard ChatML template (used by Qwen, Hermes, and fine-tunes)",
        supported_architectures=("qwen", "qwen2", "qwen2.5", "chatml", "hermes", "yi"),
        canonical_source=(
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        ),
        key_delimiters=("<|im_start|>", "<|im_end|>"),
        expected_roles=("system", "user", "assistant"),
    ),
    "llama2": CanonicalReference(
        name="Llama-2-Chat",
        description="Standard Llama-2 chat template with [INST] tags and <<SYS>> wrapper",
        supported_architectures=("llama", "llama2", "llama-2"),
        canonical_source=(
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + ' ' }}"
            "{% endif %}"
            "{% endfor %}"
        ),
        key_delimiters=("[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"),
        expected_roles=("system", "user", "assistant"),
    ),
    "llama3": CanonicalReference(
        name="Llama-3-Instruct",
        description="Standard Llama-3 header-based instruction template",
        supported_architectures=("llama", "llama3", "llama-3", "llama-3.1", "llama-3.2", "llama-3.3"),
        canonical_source=(
            "{{- bos_token }}"
            "{%- for message in messages %}"
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}"
            "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{%- endif %}"
        ),
        key_delimiters=("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"),
        expected_roles=("system", "user", "assistant"),
    ),
    "mistral": CanonicalReference(
        name="Mistral-Instruct",
        description="Standard Mistral instruction template with [INST] delimiter and alternating turns",
        supported_architectures=("mistral", "mixtral", "zephyr"),
        canonical_source=(
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        ),
        key_delimiters=("[INST]", "[/INST]"),
        expected_roles=("user", "assistant"),
    ),
    "gemma": CanonicalReference(
        name="Gemma-it",
        description="Standard Gemma instruction template with turn tokens",
        supported_architectures=("gemma", "gemma2", "gemma-2", "gemma-3"),
        canonical_source=(
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{{'<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn>\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<start_of_turn>model\\n'}}"
            "{% endif %}"
        ),
        key_delimiters=("<start_of_turn>", "<end_of_turn>"),
        expected_roles=("user", "model"),
    ),
    "phi3": CanonicalReference(
        name="Phi-3-Instruct",
        description="Standard Phi-3 instruction template with pipe tags",
        supported_architectures=("phi3", "phi-3", "phi-3.5", "phi4", "phi-4"),
        canonical_source=(
            "{% for message in messages %}"
            "{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|assistant|>\\n'}}"
            "{% endif %}"
        ),
        key_delimiters=("<|end|>", "<|user|>", "<|assistant|>"),
        expected_roles=("system", "user", "assistant"),
    ),
}


def _normalize(template_text: str) -> str:
    """Normalize whitespace and strip Jinja comments for structural comparison."""
    # Remove Jinja comments {# ... #}
    no_comments = re.sub(r"\{#.*?#\}", "", template_text, flags=re.DOTALL)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", no_comments).strip()


# Pre-compute normalized forms of canonical sources (they are constants).
_NORMALIZED_REFS: dict[str, str] = {
    key: _normalize(ref.canonical_source)
    for key, ref in CANONICAL_REFERENCES.items()
}

# Length-ratio threshold: if the input is more than this factor longer/shorter
# than a reference, skip the expensive SequenceMatcher comparison entirely.
_LENGTH_RATIO_CUTOFF = 5.0


@dataclass(slots=True)
class ComparisonResult:
    matched_reference: CanonicalReference | None = None
    similarity_score: float = 0.0
    expected_reference: CanonicalReference | None = None
    architecture_divergence: bool = False
    evidence: dict[str, object] = field(default_factory=dict)


@lru_cache(maxsize=1)
def pinned_references() -> tuple[dict[str, object], ...]:
    """Load packaged snapshots only after verifying their manifest digests."""
    directory = Path(__file__).resolve().parents[1] / "data" / "templates"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for record in manifest["references"]:
        filename = record["filename"]
        if Path(filename).name != filename:
            raise ValueError("Reference filename must stay within the packaged corpus")
        raw = (directory / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Pinned reference digest mismatch: {filename}")
        result.append({**record, "content": raw.decode("utf-8")})
    return tuple(result)


def compare_template_with_references(
    template_text: str,
    declared_architecture: str | None = None,
    threshold: float = 0.90,
) -> ComparisonResult:
    """Report heuristic resemblance independently of exact snapshot equality."""
    if len(template_text) > limits.MAX_TEMPLATE_BYTES:
        return ComparisonResult(evidence={"status": "skipped", "reason": "template byte budget"})
    raw = template_text.encode("utf-8", errors="replace")
    if len(raw) > limits.MAX_TEMPLATE_BYTES:
        return ComparisonResult(evidence={"status": "skipped", "reason": "template byte budget"})
    norm_input = _normalize(template_text)
    input_len = len(norm_input)
    arch = (declared_architecture or "").lower().strip()

    valid_refs_for_arch = [
        ref for ref in CANONICAL_REFERENCES.values()
        if arch and arch in ref.supported_architectures
    ]
    expected_ref: CanonicalReference | None = valid_refs_for_arch[0] if valid_refs_for_arch else None

    best_match: CanonicalReference | None = None
    best_score: float = 0.0

    for key, ref in CANONICAL_REFERENCES.items():
        norm_ref = _NORMALIZED_REFS[key]
        ref_len = len(norm_ref)

        # Fast length-ratio pre-filter: skip obviously dissimilar templates
        # to avoid O(n²) SequenceMatcher on very large inputs.
        if ref_len and input_len:
            length_ratio = max(input_len, ref_len) / min(input_len, ref_len)
            if length_ratio > _LENGTH_RATIO_CUTOFF:
                continue

        # Delimiters are weak format hints. They never boost the similarity
        # score or establish provenance, identity, or safety.
        ratio = difflib.SequenceMatcher(None, norm_input, norm_ref).ratio()

        if ratio > best_score:
            best_score = ratio
            best_match = ref

    matched = best_match if best_score >= threshold else None
    divergence = False

    if valid_refs_for_arch and matched is not None:
        if matched not in valid_refs_for_arch:
            divergence = True

    evidence: dict[str, object] = {
        "best_match": best_match.name if best_match else None,
        "score": round(best_score, 3),
        "threshold": threshold,
        "declared_architecture": declared_architecture,
        "expected_reference": expected_ref.name if expected_ref else None,
        "comparison_kind": "format resemblance; not authentication",
        "architecture_is_untrusted_metadata": True,
        "exact_reference": None,
    }
    digest = hashlib.sha256(raw).hexdigest()
    for record in pinned_references():
        if digest == record["sha256"] and template_text == record["content"]:
            evidence["exact_reference"] = {key: value for key, value in record.items() if key != "content"}
            break

    return ComparisonResult(
        matched_reference=matched,
        similarity_score=best_score,
        expected_reference=expected_ref,
        architecture_divergence=divergence,
        evidence=evidence,
    )

