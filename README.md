# ModelGuard

Static inspection of GGUF/safetensors containers and associated chat templates.
Implemented modules: **File Parser**, **Structure Analyser**, and **Template
Analyser**. ModelGuard does not load tensor values, run inference, render or
evaluate template expressions, or authenticate model weights. Findings are
static indicators and do not establish malicious intent or prove exploitability.

## Install and run

Python 3.11+ is required. Container parsing uses the standard library; template
inspection requires Jinja 3.1.6+ within major version 3.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m modelguard scan samples/valid_model.gguf
.venv/bin/python -m modelguard scan path/to/model.safetensors \
  --json report.json \
  --html report.html
.venv/bin/python demo_template.py
.venv/bin/python demo_template.py path/to/model.gguf
.venv/bin/python -m pytest -q
.venv/bin/python -m modelguard.evaluate_templates
```

`modelguard scan` runs all three modules. The demo's file mode runs parsing and
Module 3 only. Exit **0** means no ERROR/CRITICAL findings and no incomplete
requested analysis; **1** means such findings or incomplete template coverage;
**2** means a report could not be written. Warnings require review. An absent
template is explicitly reported and is not evidence of safety.

## Sources and loader precedence

Supported sources: embedded `tokenizer.chat_template`, named
`tokenizer.chat_template.<name>`, root `chat_template`, legacy
`tokenizer_config.json`, `processor_config.json` and `chat_template.json`, modern
`chat_template.jinja`, and `additional_chat_templates/<name>.jinja`.

Every discovered alternative is inspected within the budgets and labelled
effective or shadowed. `--template-profile auto` selects embedded GGUF precedence
for GGUF and Transformers tokenizer precedence for safetensors. Explicit options
are `gguf`, `transformers`, and `processor`. Modern Jinja overrides legacy config
in the Transformers profile; processor mode also reports incompatible mixtures
of legacy JSON and modern files. Arbitrary application-level overrides cannot be
inferred. See the [primary Transformers documentation](https://huggingface.co/docs/transformers/en/chat_templating_writing#storing-and-loading-chat-templates).

Sidecars must be regular UTF-8 files. Reads are bounded on the opened descriptor.
Within-package symlinks are accepted; external links, including some Hugging Face
cache layouts, are explicitly skipped. Use a self-contained package for complete
sidecar coverage. Malformed JSON, duplicate keys, invalid schemas, unreadable
files and budget failures are reported rather than silently ignored.

## Checks and interpretation

| Codes | Meaning |
| --- | --- |
| CT101/102/104 | Potential object-internals, code, process or file access in executable expressions; exploitability is unverified. |
| CT103/105 | Unresolved dynamic attributes; external inclusion, imports or inheritance. |
| CT110–112 | Conservative role-filtering, suspicious-output and Unicode-control advisories. |
| CT113 | Generation-format compatibility notice; some formats intentionally omit the flag. |
| CT120/126 | Malformed syntax versus unsupported dialect. Trusted generation and loop-control extensions are supported. |
| CT121/122/124/125 | Resource limits, unavailable parser or failed/timed-out worker. |
| CT130–132/134 | Format resemblance, architecture advisory, custom format, or exact pinned snapshot equality. |
| CT140–142 | Extraction failures and source conflicts. |

Code checks resolve bounded constant strings and assignment aliases; inert
comments and plain text are not interpreted as executable code. Prompt checks
inspect potential output, including bounded concatenation and normalized Unicode.
Role and prompt analysis remain heuristics; arbitrary runtime-dependent text,
context and multilingual attacks cannot be exhaustively resolved statically.

The historical `CANONICAL_REFERENCES` constant contains hand-written format
examples for API compatibility, with unverified provenance. Separately packaged
llama.cpp snapshots have source URLs, model labels, revisions and SHA-256 digests.
CT134 requires exact bytes. Neither resemblance nor snapshot equality
authenticates a publisher, model weights, runtime or safety.

## Resource enforcement and coverage

Central limits: 256 KiB/template, 2 MiB total template text, 32 templates,
2 MiB/JSON sidecar, 4 MiB total sidecar reads, 256 named-directory entries,
20,000 tokens, 10,000 AST nodes and syntax/AST depth 50. Oversized inputs stop
before expensive stages; iterative AST inventory precedes inspection.
Constant-string expansion is bounded too; exceeding 32 candidate alias paths
produces incomplete coverage instead of silently discarding alternatives.

Inspection runs in an isolated Python worker with a 2-second wall deadline per
template and an 8-second aggregate inspection deadline. POSIX adds a CPU limit;
Linux adds a 512 MiB address-space limit. Worker startup contributes to latency.
No model-directory code or model-provided extension is imported.

Terminal, JSON, HTML and demo output separate findings from coverage: completed,
absent, unavailable, unsupported, skipped, failed or incomplete. Per-template
records retain source, name, effective status, digest, AST summary, timing and
available worker peak RSS. Complete ASTs and lexical token streams are not saved.
Incomplete container parsing also makes template coverage incomplete.

## Verification

Pytest reports the current count; guides intentionally avoid fixed counts and
runtime promises. CI runs Python 3.11 and 3.13. Regression tests exercise discovery,
precedence, inert text, equivalent access forms, aliases, Unicode, dialects,
resource failures, reporting, and controls that verify templates are not run.

The offline evaluation uses five pinned upstream templates, two held-out upstream
templates in a source checkout, and nine synthetic indicator cases. It reports
precision, recall, false-positive rate, latency and worker peak RSS separately for
code and prompt indicators. These small fixture results are reproducible, not
real-world accuracy estimates or exploit validation. Broader claims require a
larger independently labelled corpus.

Tiny synthetic container samples can be regenerated with
`python3 samples/generate_samples.py`; they are not inference-ready models.
Files in `review_artifacts/` are historical Module 1/2 demonstrations. Current
guides are in `docs/`; `python3 scripts/update_guides.py` refreshes the three Word
guides. AFL++ parser fuzzing and final lifecycle verdict reporting remain future
work.
