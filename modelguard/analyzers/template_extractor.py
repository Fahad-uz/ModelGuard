"""Bounded package discovery with explicit diagnostics and loader precedence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from itertools import islice

from modelguard import limits
from modelguard.models.results import Finding, ParsedModel


@dataclass(slots=True)
class ExtractedTemplate:
    source: str
    content: str
    name: str = "default"
    effective: bool = True
    priority: int = 0


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key[:100]}")
        result[key] = value
    return result


class TemplateExtractor:
    """Discover every supported source; mark the selected template per name.

    GGUF uses its embedded tokenizer templates. The Transformers profile gives
    modern files priority over legacy configuration. All discovered alternatives
    are inspected too, with their effective/shadowed status retained.
    """

    def extract(self, model: ParsedModel, *, profile: str = "auto") -> list[ExtractedTemplate]:
        if profile not in {"auto", "gguf", "transformers", "processor"}:
            raise ValueError("Unknown template loader profile")
        selected = ("gguf" if model.format == "GGUF" else "transformers") if profile == "auto" else profile
        state: dict[str, object] = {
            "status": "pending", "complete": True, "loader_profile": selected,
            "inspected_sources": [], "skipped_sources": [], "conflicts": [],
            "extraction_findings": [], "bytes_read": 0, "template_bytes": 0,
        }
        model.extra["template_analysis"] = state
        model.extra["chat_templates"] = []
        templates: list[ExtractedTemplate] = []

        def issue(code: str, source: str, message: str, *, severity: str = "ERROR") -> None:
            incomplete = severity == "ERROR"
            if code == "CT121":
                severity = "WARNING"
                incomplete = True
            if len(state["extraction_findings"]) < 64:
                state["extraction_findings"].append(asdict(Finding(  # type: ignore[union-attr,arg-type]
                    code, severity, "template_syntax" if code == "CT121" else "template_extraction", message, source[:4096],
                    {"source": source[:4096]}, "Review the source or explicitly choose the intended loader profile.")))
            if incomplete:
                state["complete"] = False
                if len(state["skipped_sources"]) < 64:
                    state["skipped_sources"].append({"source": source[:4096], "reason": message})  # type: ignore[union-attr]

        def add(source: str, text: object, name: object = "default", priority: int = 0) -> None:
            if not isinstance(text, str) or not isinstance(name, str) or not name or len(name) > 4096 or len(source) > 4096:
                issue("CT140", source, "Template content and a non-empty name must be strings")
                return
            byte_len = len(text.encode("utf-8", errors="replace"))
            if len(templates) >= limits.MAX_TEMPLATES:
                issue("CT121", source, "Template count limit exceeded; source was not analysed")
                return
            if byte_len > limits.MAX_TEMPLATE_BYTES:
                issue("CT121", source, "Template byte limit exceeded; source was not parsed")
                return
            if int(state["template_bytes"]) + byte_len > limits.MAX_TEMPLATE_TOTAL_BYTES:
                issue("CT121", source, "Aggregate template byte limit exceeded")
                return
            state["template_bytes"] = int(state["template_bytes"]) + byte_len
            templates.append(ExtractedTemplate(source, text, name, priority=priority))

        metadata = model.metadata
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if not isinstance(key, str):
                    continue
                if key == "tokenizer.chat_template":
                    add("metadata.tokenizer.chat_template", value, priority=100 if selected == "gguf" else 0)
                elif key.startswith("tokenizer.chat_template."):
                    add(f"metadata.{key}", value, key[len("tokenizer.chat_template."):], 100 if selected == "gguf" else 0)
                elif key == "chat_template":
                    add("metadata.chat_template", value, priority=0)

        directory = model.path.parent.resolve()

        def present(path: Path) -> bool:
            try:
                path.lstat()
                return True
            except FileNotFoundError:
                return False
            except OSError as exc:
                issue("CT140", path.name, f"Cannot inspect template source: {type(exc).__name__}")
                return False

        def read_source(path: Path, cap: int) -> str | None:
            source = str(path.relative_to(directory))
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(directory):
                    raise ValueError("Template source resolves outside the model package")
                flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(resolved, flags)
                with os.fdopen(fd, "rb") as stream:
                    facts = os.fstat(stream.fileno())
                    if not stat.S_ISREG(facts.st_mode):
                        raise ValueError("Template source must be a regular file")
                    remaining = limits.MAX_TEMPLATE_SOURCE_BYTES - int(state["bytes_read"])
                    effective_cap = min(cap, remaining)
                    if facts.st_size > effective_cap:
                        raise ValueError("Template source exceeds its file or aggregate byte budget")
                    raw = stream.read(effective_cap + 1)
                if len(raw) > effective_cap:
                    raise ValueError("Template source grew beyond its read budget")
                state["bytes_read"] = int(state["bytes_read"]) + len(raw)
                text = raw.decode("utf-8")
                state["inspected_sources"].append({  # type: ignore[union-attr]
                    "source": source, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                })
                return text
            except (OSError, ValueError, UnicodeError) as exc:
                issue("CT140", source, f"Cannot read template source: {str(exc)[:240]}")
                return None

        def config(path: Path, priority: int) -> None:
            if not present(path):
                return
            raw = read_source(path, limits.MAX_TEMPLATE_CONFIG_BYTES)
            if raw is None:
                return
            try:
                data = json.loads(raw, object_pairs_hook=_unique_object,
                                  parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(data, dict):
                    raise ValueError("Template configuration must be an object")
                if "chat_template" not in data:
                    return
                value = data["chat_template"]
                source = f"{path.name}#chat_template"
                if isinstance(value, str):
                    add(source, value, priority=priority)
                elif isinstance(value, list):
                    seen: set[str] = set()
                    for item in value[:limits.MAX_TEMPLATES + 1]:
                        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                            issue("CT140", source, "Named template entries require string name and template fields")
                            continue
                        name = item["name"]
                        if name in seen:
                            issue("CT140", source, f"Duplicate template name: {name[:100]}")
                            continue
                        seen.add(name)
                        add(f"{path.name}#{name}", item.get("template"), name, priority)
                    if len(value) > limits.MAX_TEMPLATES:
                        issue("CT121", source, "Named template count exceeds configured limit")
                elif isinstance(value, dict):
                    for name, text in islice(value.items(), limits.MAX_TEMPLATES + 1):
                        add(f"{path.name}#{name}", text, name, priority)
                    if len(value) > limits.MAX_TEMPLATES:
                        issue("CT121", source, "Named template count exceeds configured limit")
                else:
                    issue("CT140", source, "Unsupported chat_template value type")
            except (ValueError, RecursionError) as exc:
                issue("CT140", path.name, f"Invalid template configuration: {str(exc)[:240]}")

        config(directory / "tokenizer_config.json", 10 if selected == "transformers" else -1)
        config(directory / "processor_config.json", 10 if selected == "processor" else -1)
        legacy = directory / "chat_template.json"
        config(legacy, 20 if selected == "processor" else -1)
        modern = directory / "chat_template.jinja"
        modern_present = present(modern)
        if modern_present:
            text = read_source(modern, limits.MAX_TEMPLATE_BYTES)
            if text is not None:
                add(modern.name, text, priority=30 if selected != "gguf" else -1)
        additional = directory / "additional_chat_templates"
        additional_present = present(additional)
        if additional_present:
            try:
                resolved = additional.resolve(strict=True)
                if not resolved.is_relative_to(directory):
                    raise ValueError("Named template directory resolves outside the model package")
                with os.scandir(resolved) as entries:
                    paths = []
                    for count, entry in enumerate(entries):
                        if count >= limits.MAX_TEMPLATE_DIRECTORY_ENTRIES:
                            issue("CT121", additional.name, "Named template directory entry budget exceeded")
                            break
                        if entry.name.endswith(".jinja"):
                            paths.append(additional / entry.name)
                for path in sorted(paths):
                    text = read_source(path, limits.MAX_TEMPLATE_BYTES)
                    if text is not None:
                        add(str(path.relative_to(directory)), text, path.stem, 40 if selected != "gguf" else -1)
            except (OSError, ValueError) as exc:
                issue("CT140", additional.name, f"Cannot inspect named templates: {str(exc)[:240]}")

        if selected == "processor" and present(legacy) and (modern_present or additional_present):
            issue("CT142", "chat_template.json", "Processor package mixes legacy JSON and modern Jinja templates")

        winners: dict[str, ExtractedTemplate] = {}
        for template in templates:
            if template.priority < 0:
                continue
            previous = winners.get(template.name)
            if previous is None or template.priority >= previous.priority:
                winners[template.name] = template
        for template in templates:
            template.effective = winners.get(template.name) is template
        for name, winner in winners.items():
            alternatives = [t for t in templates if t.name == name and t is not winner]
            if alternatives and any(t.content != winner.content for t in alternatives):
                conflict = {"name": name, "effective_source": winner.source,
                            "shadowed_sources": [t.source for t in alternatives]}
                state["conflicts"].append(conflict)  # type: ignore[union-attr]
                issue("CT141", winner.source, "Conflicting templates discovered; loader precedence selects one source", severity="WARNING")
        state["status"] = "pending" if templates else ("absent" if state["complete"] else "incomplete")
        return templates
