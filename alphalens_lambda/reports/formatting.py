import html
import json
import re
from typing import Any

from alphalens_lambda.reports.schemas import ReportSection


def build_sections_text(sections: list[ReportSection]) -> str:
    output = "The user requested the following report sections:\n\n"
    for section in sorted(sections, key=lambda item: item.order):
        output += f"Section: {section.title}\n"
        output += f"Description: {section.description}\n"
        if section.user_notes and section.user_notes.strip():
            output += f"User Notes: {section.user_notes}\n"
        output += "\n"
    return output.strip()


def json_to_readable_string(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = "  " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                nested = json_to_readable_string(item, indent + 1)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{prefix}{key}: {item}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = json_to_readable_string(item, indent + 1).splitlines()
                if rendered:
                    lines.append(f"{prefix}- {rendered[0].lstrip()}")
                    lines.extend(rendered[1:])
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{value}")
    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def extract_report_envelope(raw: str | dict[str, Any]) -> dict[str, Any]:
    parsed = raw if isinstance(raw, dict) else extract_json_object(raw)
    if parsed:
        nested = parsed.get("message", {}).get("content", {})
        content = nested.get("content")
        citations = nested.get("citations", [])
        if isinstance(content, str) and content.strip():
            return {
                "message": {
                    "content": {
                        "content": content.strip(),
                        "citations": citations if isinstance(citations, list) else [],
                    }
                }
            }
        direct = parsed.get("content")
        if isinstance(direct, str) and direct.strip():
            return {
                "message": {
                    "content": {
                        "content": direct.strip(),
                        "citations": parsed.get("citations", []),
                    }
                }
            }
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=True)
    return {"message": {"content": {"content": text.strip(), "citations": []}}}


def normalize_html(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:html)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start_marker = "<html><body>"
    end_marker = "</body></html>"
    start = cleaned.find(start_marker)
    end = cleaned.rfind(end_marker)
    if start == -1 or end == -1:
        raise ValueError("HTML builder did not return the required standalone HTML envelope")
    normalized = cleaned[start : end + len(end_marker)]
    if not normalized.startswith(start_marker) or not normalized.endswith(end_marker):
        raise ValueError("Invalid HTML email envelope")
    return normalized


def fallback_html(report_text: str) -> str:
    paragraphs = "".join(
        f"<p style=\"margin:0 0 12px;line-height:1.5\">{html.escape(line)}</p>"
        for line in report_text.splitlines()
        if line.strip()
    )
    return f"<html><body>{paragraphs}</body></html>"


def short_error(error: Exception, limit: int = 240) -> str:
    message = str(error).strip() or error.__class__.__name__
    return message[:limit]
