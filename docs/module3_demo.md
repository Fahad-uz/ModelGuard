# ModelGuard Module 3 Demonstration Guide

## Scope

Module 3 discovers embedded and package chat templates, then inspects bounded Jinja syntax without rendering, compiling or evaluating template expressions. Findings are static indicators; exploitability depends on the actual loader, sandbox and context. Do not claim absolute risk elimination for software processing hostile input.

## Demonstration

Run these commands from the repository root on a branch containing Module 3:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest tests/test_template_analyzer.py tests/test_template_regressions.py -q
.venv/bin/python demo_template.py
.venv/bin/python -m modelguard.evaluate_templates
```

Pytest supplies the current counts and timings. The demo shows a benign format example, suspicious code constructs, possible role filtering with suspicious output text, and malformed syntax with incomplete coverage. No template is rendered or executed.

## Reviewer explanations

Code-risk, prompt-risk, compatibility, reference and coverage findings are separate. Code syntax does not prove a sandbox escape. Prompt phrases and role filtering are conservative heuristics requiring context. Intentional system-role rejection is informational.

CT130 means resemblance to a hand-written format example; it has unverified provenance. CT134 requires exact bytes from a pinned upstream snapshot and includes a model label, source URL, revision and SHA-256. Neither authenticates model weights or proves safety.

Retained information includes source, name, effective status, digest, AST node-type summary, depth/count, coverage, timings and available worker peak RSS. Full ASTs and lexical token streams are not retained.

Absent means supported discovery found no template. Incomplete means at least one source could not be inspected within the supported syntax and resource limits. Show both findings and coverage when presenting results.

The offline evaluation uses independent upstream templates and explicitly synthetic indicator cases. Its precision, recall, false-positive rate, latency and memory results apply to that small corpus, not all model packages. Runtime-dependent behavior and verified exploitation remain outside this module's scope.
