"""Refresh the Word guides from Markdown using standard-library OOXML/ZIP."""

from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", NS)
GUIDES = {
    "docs/module3_demo.md": "ModelGuard_Module3_Demonstration_Guide.docx",
    "docs/execution_commands.md": "ModelGuard_Execution_Commands.docx",
    "docs/run_guide.md": "ModelGuard_Execution_and_Run_Guide.docx",
}


def refresh(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination) as archive:
        parts = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    document = ET.fromstring(parts["word/document.xml"])
    body = document.find(f"{{{NS}}}body")
    section = body.find(f"{{{NS}}}sectPr")
    body.clear()
    code = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            code = not code
            continue
        paragraph = ET.SubElement(body, f"{{{NS}}}p")
        if line.startswith("#") and not code:
            level = len(line) - len(line.lstrip("#"))
            properties = ET.SubElement(paragraph, f"{{{NS}}}pPr")
            ET.SubElement(properties, f"{{{NS}}}pStyle", {f"{{{NS}}}val": "Title" if level == 1 else "Heading1"})
            line = line.lstrip("# ")
        run = ET.SubElement(paragraph, f"{{{NS}}}r")
        if code:
            properties = ET.SubElement(run, f"{{{NS}}}rPr")
            ET.SubElement(properties, f"{{{NS}}}rFonts", {f"{{{NS}}}ascii": "Consolas", f"{{{NS}}}hAnsi": "Consolas"})
        text = ET.SubElement(run, f"{{{NS}}}t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
        text.text = line
    if section is not None:
        body.append(section)
    parts["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    temporary = destination.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    temporary.replace(destination)


if __name__ == "__main__":
    for source, destination in GUIDES.items():
        refresh(ROOT / source, ROOT / destination)
        print(f"Updated {destination}")
