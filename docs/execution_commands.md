# ModelGuard Execution Commands

Use Python 3.11+ and the repository branch containing Module 3. Run from the repository root.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

## Template-only inspection

```bash
.venv/bin/python demo_template.py
.venv/bin/python demo_template.py path/to/model.gguf
.venv/bin/python demo_template.py path/to/model.safetensors
```

File mode uses parsing and Module 3, without the structural analyser. Review parser issues and template coverage.

## Full scanner

```bash
.venv/bin/python -m modelguard scan samples/valid_model.gguf
.venv/bin/python -m modelguard scan path/to/model.safetensors \
  --json report.json \
  --html report.html
.venv/bin/python -m modelguard scan path/to/model.safetensors --template-profile processor
```

Profiles: auto, gguf, transformers and processor. Auto uses embedded GGUF or Transformers tokenizer precedence. All discovered alternatives are inspected and labelled effective or shadowed. Processor mode reports incompatible mixed legacy/modern storage.

## Tests and evaluation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_template_analyzer.py tests/test_template_regressions.py -q
.venv/bin/python -m modelguard.evaluate_templates
```

Read current counts from pytest. Evaluation reports scoped fixture metrics, latency and available worker peak RSS; it does not establish real-world accuracy or prove exploits.

## Exit codes

0: no ERROR/CRITICAL findings and no incomplete requested analysis. 1: such findings or incomplete template coverage. 2: report-writing failure. Warnings require review; an absent template is not evidence of safety. Reports are still written when findings cause exit 1.

Open report.html in a browser or file manager. macOS: open report.html. Linux: xdg-open report.html.
