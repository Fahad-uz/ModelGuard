# ModelGuard Setup and Run Guide

## Select the project

Open a terminal, enter your ModelGuard directory, and check that the selected branch contains the Module 3 fixes. Earlier main-branch revisions may contain only Modules 1 and 2.

```bash
cd ~/Documents/ModelGuard
git status --short --branch
```

## Create the environment

No pre-existing virtual environment is assumed. Direct interpreter paths make activation optional.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

## Verify and demonstrate

```bash
.venv/bin/python -m pytest -q
.venv/bin/python demo_template.py
.venv/bin/python -m modelguard.evaluate_templates
```

Use the counts printed by pytest. The demonstration is synthetic; the evaluation also includes independent pinned and held-out templates, with explicitly scoped accuracy, latency and worker memory metrics.

## Scan a model package

Keep the model and its tokenizer/processor files together. Complete coverage requires a self-contained package: external symlinks are explicitly skipped.

```bash
.venv/bin/python -m modelguard scan path/to/model.safetensors \
  --template-profile transformers \
  --json report.json \
  --html report.html
```

Use processor for a multimodal processor, gguf for embedded GGUF precedence, or auto for the format-based default. Application-level template overrides cannot be inferred automatically.

## Interpret and view results

Read findings and coverage together. Completed means supported static checks finished within budget. Absent means no supported template was discovered. Incomplete means some source was invalid, unavailable, unsupported, over budget or failed inspection. Neither absence nor completion guarantees safety.

Code findings are potential-risk syntax. Prompt and role findings are heuristics. Format resemblance differs from exact pinned snapshot equality. None authenticates model weights or proves runtime exploitability. No template is rendered or evaluated.

Open report.html in a browser: open report.html on macOS, or xdg-open report.html on Linux. Exit 1 reports ERROR/CRITICAL findings or incomplete template coverage while still writing reports; exit 2 indicates report-writing failure.
