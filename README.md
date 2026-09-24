# ModelGuard

**ModelGuard: A Unified Security Toolkit for the ML Model Supply Chain** is a B.Tech cybersecurity project. This First Review milestone implements **Module 1, static file parsing**, and **Module 2, structural anomaly detection** for GGUF and safetensors containers. It inspects model headers, metadata, and tensor descriptors without running inference or interpreting executable content.

ModelGuard reports specific structural problems. A malformed file is **not** automatically classified as malware, and this milestone does not provide a SAFE/SUSPICIOUS/MALICIOUS verdict.

## Implemented modules

| Module | What it does |
| --- | --- |
| File Parser | Detects GGUF or safetensors and returns normalized file facts, metadata, tensor shapes, types, offsets, and controlled parser issues. |
| Structure Analyser | Checks bounds, declared sizes, types, counts, alignment, layout, duplicates, truncation, and resource-exhaustion indicators; returns structured findings. |

The terminal command, JSON export, and HTML page are presentation views for these two modules. The HTML page is a First Review aid, not the planned final reporting module.

## Requirements and installation

Python 3.11 or newer is required. Runtime parsing has no third-party dependencies. From this repository's root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

The optional `test` extra installs pytest. If your Python installation does not provide pip in virtual environments, create the environment with [uv](https://docs.astral.sh/uv/) and install with `uv pip install --python .venv/bin/python -e '.[test]'`.

## Scan a model

```bash
.venv/bin/python -m modelguard scan samples/valid_model.safetensors
.venv/bin/python -m modelguard scan samples/valid_model.gguf
.venv/bin/python -m modelguard scan samples/invalid_offset.gguf \
  --json review_artifacts/invalid_offset_gguf.json \
  --html review_artifacts/invalid_offset_gguf.html
```

The installed `modelguard scan ...` command is equivalent to `python -m modelguard scan ...`. The command returns exit code **0** when there are no error-level findings, **1** when the scan reports an error or critical finding, and **2** if an output file cannot be written. An error exit still produces the report.

Example of a valid scan:

```text
ModelGuard review
=================
File: samples/valid_model.safetensors
Format: SAFETENSORS
Size: 199 bytes
Parser: OK (0 errors, 0 warnings)
Findings: 0
Header length: 179
Data start: 187
Declared tensors: 2
Declared metadata: 2

Metadata (2):
  source [str]: synthetic
  purpose [str]: first review

Tensors (2):
  weight: dtype=F32, shape=[2], offsets=[0, 8]
  bias: dtype=F16, shape=[2], offsets=[8, 12]

Findings (0):
  No structural violations detected.
```

JSON contains the normalized parser result, parser issues, tensor descriptors, structural findings, and counts by severity. HTML is self-contained and escapes model-provided text before displaying it.

## Supported formats and features

### Safetensors

The parser reads the 8-byte little-endian header length and bounded UTF-8 JSON header. It extracts `__metadata__`, tensor names, dtypes, shapes, offsets, and the data-region start. It rejects duplicate JSON keys and invalid JSON, including `NaN`/`Infinity`. The analyser checks dtype and shape validity, calculated byte sizes (including sub-byte dtypes), offset bounds, gaps, overlaps, and trailing data. Empty tensors and scalar shapes are handled.

### GGUF

The parser supports **GGUF versions 2 and 3**, including little-endian v2/v3 and big-endian v3. It reads magic, version, counts, metadata types **0–12** (bounded strings and arrays), and tensor descriptors. It computes the aligned tensor data start and uses known GGML block sizes to estimate each tensor's byte range. The analyser checks counts, names, dimensions, alignment, packed offsets, type/shape size consistency, and file bounds. Unknown or deprecated GGML tensor types receive explicit issues instead of guessed sizes. Supported type IDs and block sizes are listed in [`modelguard/parsers/ggml_types.py`](modelguard/parsers/ggml_types.py).

Implementation follows the primary [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [llama.cpp GGUF reader](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp), and [GGML type definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h). Safetensors checks follow its [format specification](https://github.com/huggingface/safetensors#format) and [reference parser](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs).

## Security design

- Reads only bounded headers, metadata, and descriptors. Tensor payloads are not loaded for inspection.
- Never deserializes pickle, evaluates templates, imports model-specified code, executes model content, or runs inference.
- Applies centralized ceilings in [`modelguard/limits.py`](modelguard/limits.py): 16 MiB parsed headers, 1 MiB strings, 100,000 metadata fields/tensors/array elements, bounded nesting, and checked size arithmetic.
- Returns controlled parser issues and structural findings for malformed input. Unexpected failures are contained at the scan boundary.
- Separates parsing facts from analysis judgments and from the CLI/HTML presentation.

## Samples, tests, and First Review

All committed sample files are tiny and synthetic. Regenerate them with:

```bash
.venv/bin/python samples/generate_samples.py
```

`valid_model.gguf` is a specification-correct minimal **container fixture**, not an inference-ready model with a real architecture. The malformed fixtures demonstrate invalid magic, truncation, and invalid offsets. No large pretrained models are committed.

Run the full suite:

```bash
.venv/bin/python -m pytest -q
```

Final local result for this milestone: **50 passed** on Python 3.11. GitHub Actions runs the suite on Python 3.11 and 3.13 for pushes and pull requests. For a 3–5 minute presentation, follow [`review_artifacts/FIRST_REVIEW_DEMO.md`](review_artifacts/FIRST_REVIEW_DEMO.md).

## Known limitations and next modules

- Static structural findings describe malformed or suspicious layout; they do not establish malicious intent or guarantee that a downstream loader is safe.
- The GGML size table covers documented current storage types; unknown/deprecated and future types are explicitly unsupported until verified against upstream layouts.
- The scanner does not inspect tensor values. It does not verify model semantics, architecture compatibility, or whether a model can run inference.
- Defensive limits may reject unusually large but legitimate metadata headers or arrays. These limits are intentionally conservative for this review build.
- The first review supports one file per scan and emits local reports only.

**Planned next:** Module 3 will statically inspect template metadata and flag risky constructs without executing Jinja or model-provided code. Later work will add the AFL++ fuzzing harness and the final reporting/verdict module.
