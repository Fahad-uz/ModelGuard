"""Private bounded worker for syntax analysis; accepts and returns JSON only."""

from __future__ import annotations

from dataclasses import asdict
import json
import sys

from modelguard import limits


def main() -> int:
    # The parent enforces a wall-clock timeout on every platform. POSIX workers
    # additionally receive a CPU limit; Linux also enforces an address-space cap.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except (ImportError, OSError, ValueError):
        pass
    from modelguard.analyzers.template_analyzer import TemplateAnalyzer
    from modelguard.analyzers.template_extractor import ExtractedTemplate

    cap = limits.MAX_TEMPLATE_WORKER_OUTPUT_BYTES * 2
    raw = sys.stdin.buffer.read(cap + 1)
    if len(raw) > cap:
        return 2
    request = json.loads(raw)
    analyzer = TemplateAnalyzer()
    if analyzer._jinja_env is None:
        return 2
    findings, summary = analyzer._inspect_single_template(
        ExtractedTemplate(**request["template"]), request.get("architecture"))
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary["worker_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    except ImportError:
        summary["worker_peak_rss_bytes"] = None
    response = json.dumps({"findings": [asdict(item) for item in findings], "summary": summary}, ensure_ascii=True)
    if len(response.encode("utf-8")) > limits.MAX_TEMPLATE_WORKER_OUTPUT_BYTES:
        return 2
    sys.stdout.write(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
