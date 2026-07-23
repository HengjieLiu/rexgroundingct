#!/usr/bin/env python3
"""Create auditable Markdown and raw-source snapshots for ReXGroundingCT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


TOOL_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
SNAPSHOTS_DIR = ROOT / "snapshots"
USER_AGENT = "ReXGroundingCT-Archive/1.0 (+https://rexrank.ai/ReXGroundingCT/)"

CATEGORY_ORDER = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
CATEGORY_LABELS = {
    "1a": "Bronchial wall thickening",
    "1b": "Bronchiectasis",
    "1c": "Emphysema (including Centrilobular, Paraseptal, Bullous)",
    "1d": "Septal thickening (including Interlobular, Reticulation)",
    "1e": "Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic)",
    "1f": "Other",
    "2a": "Linear (including subsegmental atelectasis, scarring, fibrosis)",
    "2b": "Atelectasis, consolidation",
    "2c": "Groundglass opacity",
    "2d": "Pulmonary nodules/masses",
    "2e": "Pleural effusion or thickening",
    "2f": "Honeycombing",
    "2g": "Pneumothorax",
    "2h": "Other",
    "unknown": "Uncategorized",
}


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    content_type: str
    charset: str | None
    etag: str | None
    last_modified: str | None
    fetched_at_utc: str
    content: bytes


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def snapshot_id_for(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def fetch_url(url: str, timeout: int = 45) -> FetchResult:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json,text/plain,text/csv;q=0.9,*/*;q=0.8",
        },
    )
    fetched_at = iso_utc(utc_now())
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
        headers = response.headers
        return FetchResult(
            requested_url=url,
            final_url=response.geturl(),
            status=getattr(response, "status", 200),
            content_type=headers.get_content_type() or "application/octet-stream",
            charset=headers.get_content_charset(),
            etag=headers.get("ETag"),
            last_modified=headers.get("Last-Modified"),
            fetched_at_utc=fetched_at,
            content=content,
        )


def decode_text(content: bytes, charset: str | None = None) -> str:
    candidates = [charset, "utf-8-sig", "utf-8", "windows-1252", "latin-1"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return content.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    raise UnicodeDecodeError("utf-8", content, 0, min(1, len(content)), "unable to decode source")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def normalize_windows_acl(path: Path) -> None:
    """Ensure files created by an approved elevated fetch inherit workspace ACLs."""
    if os.name != "nt":
        return
    completed = subprocess.run(
        ["icacls", str(path), "/inheritance:e", "/T", "/C", "/Q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown icacls error"
        raise OSError(f"Could not enable inherited workspace permissions for {path}: {detail}")


def load_registry(root: Path = ROOT) -> dict[str, Any]:
    registry = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported sources.json schema_version")
    return registry


def validate_fetch(source: dict[str, Any], result: FetchResult) -> None:
    if result.status != 200:
        raise ValueError(f"{source['id']}: HTTP {result.status}")
    if not result.content:
        raise ValueError(f"{source['id']}: empty response")
    text = decode_text(result.content, result.charset)
    haystack = text
    if source.get("kind") == "html":
        visible_text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        haystack = text + "\n" + visible_text
    missing = [needle for needle in source.get("validate_contains", []) if needle not in haystack]
    if missing:
        raise ValueError(f"{source['id']}: missing required source text: {missing}")


def manifest_entry(source: dict[str, Any], result: FetchResult) -> dict[str, Any]:
    return {
        "id": source["id"],
        "category": source.get("category", "source"),
        "required": bool(source.get("required", True)),
        "requested_url": sanitize_url(result.requested_url),
        "final_url": sanitize_url(result.final_url),
        "snapshot_path": source["snapshot_path"].replace("\\", "/"),
        "http_status": result.status,
        "content_type": result.content_type,
        "charset": result.charset,
        "etag": result.etag,
        "last_modified": result.last_modified,
        "fetched_at_utc": result.fetched_at_utc,
        "bytes": len(result.content),
        "sha256": sha256_bytes(result.content),
    }


def escape_md_cell(value: Any) -> str:
    if value is None or value == "":
        return "--"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def markdown_table(headers: Iterable[Any], rows: Iterable[Iterable[Any]]) -> str:
    header_values = [escape_md_cell(value) for value in headers]
    lines = [
        "| " + " | ".join(header_values) + " |",
        "| " + " | ".join("---" for _ in header_values) + " |",
    ]
    for row in rows:
        values = [escape_md_cell(value) for value in row]
        if len(values) < len(header_values):
            values.extend(["--"] * (len(header_values) - len(values)))
        lines.append("| " + " | ".join(values[: len(header_values)]) + " |")
    return "\n".join(lines)


def clean_markdown(value: str) -> str:
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    output: list[str] = []
    blank = 0
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
        if line or in_fence:
            blank = 0
            output.append(line)
        else:
            blank += 1
            if blank <= 2:
                output.append("")
    return "\n".join(output).strip() + "\n"


class MarkdownConverter:
    """Small deterministic converter for source-faithful archival Markdown."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def convert(self, content: bytes, selector: str | None = None) -> str:
        soup = BeautifulSoup(decode_text(content), "html.parser")
        for node in soup.find_all(["script", "style", "noscript", "template", "svg", "canvas"]):
            node.decompose()
        root: Tag | BeautifulSoup = soup.select_one(selector) if selector else (soup.body or soup)
        if root is None:
            root = soup.body or soup
        return clean_markdown(self._children(root))

    def _children(self, node: Tag | BeautifulSoup, depth: int = 0) -> str:
        return "".join(self._node(child, depth) for child in node.children)

    def _inline(self, node: Tag) -> str:
        value = self._children(node).strip()
        return re.sub(r"[ \t\f\v]+", " ", value)

    def _node(self, node: Any, depth: int = 0) -> str:
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            if node.parent and node.parent.name in {"pre", "code"}:
                return str(node)
            return re.sub(r"\s+", " ", str(node))
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = self._inline(node)
            return f"\n\n{'#' * int(name[1])} {text}\n\n" if text else ""
        if name == "p":
            text = self._inline(node)
            return f"\n\n{text}\n\n" if text else ""
        if name == "br":
            return "\n"
        if name == "hr":
            return "\n\n---\n\n"
        if name in {"strong", "b"}:
            text = self._inline(node)
            return f"**{text}**" if text else ""
        if name in {"em", "i"}:
            text = self._inline(node)
            return f"*{text}*" if text else ""
        if name == "del":
            text = self._inline(node)
            return f"~~{text}~~" if text else ""
        if name == "a":
            text = self._inline(node) or node.get("href", "")
            href = urljoin(self.base_url, node.get("href", ""))
            if urlsplit(href).scheme.lower() in {"data", "javascript", "vbscript"}:
                return text
            return f"[{text}]({href})" if href else text
        if name == "img":
            src = urljoin(self.base_url, node.get("src", ""))
            alt = node.get("alt", "")
            return f"![{alt}]({src})" if src else ""
        if name == "code" and node.parent and node.parent.name != "pre":
            text = node.get_text()
            fence = "``" if "`" in text else "`"
            return f"{fence}{text}{fence}"
        if name == "pre":
            text = node.get_text("", strip=False).strip("\n")
            language = ""
            code = node.find("code")
            if code:
                for class_name in code.get("class", []):
                    if class_name.startswith("language-"):
                        language = class_name.removeprefix("language-")
                        break
            return f"\n\n```{language}\n{text}\n```\n\n"
        if name in {"ul", "ol"}:
            return self._list(node, depth)
        if name == "li":
            return self._inline(node)
        if name == "table":
            return f"\n\n{self._table(node)}\n\n"
        if name == "blockquote":
            text = clean_markdown(self._children(node)).strip()
            return "\n\n" + "\n".join(f"> {line}" if line else ">" for line in text.splitlines()) + "\n\n"
        if name == "input":
            if node.get("type", "text").lower() == "hidden":
                return ""
            label = node.get("placeholder") or node.get("aria-label") or node.get("name") or node.get("type", "text")
            return f"[Input: {label}]"
        if name == "textarea":
            label = node.get("placeholder") or node.get("aria-label") or node.get("name") or "text"
            return f"[Textarea: {label}]"
        if name == "button":
            text = self._inline(node) or node.get("aria-label")
            return f"[Button: {text}]" if text else ""
        if name == "select":
            label = node.get("aria-label") or node.get("name") or node.get("id") or "select"
            options = [option.get_text(" ", strip=True) for option in node.find_all("option")]
            suffix = ": " + ", ".join(option for option in options if option) if options else ""
            return f"[Select: {label}{suffix}]"
        if name == "option":
            return ""
        if name == "dt":
            text = self._inline(node)
            return f"\n\n**{text}**\n" if text else ""
        if name == "dd":
            text = self._inline(node)
            return f"\n{text}\n" if text else ""

        rendered = self._children(node, depth)
        if name in {"article", "aside", "div", "footer", "form", "header", "main", "nav", "section", "tr"}:
            return f"\n{rendered}\n"
        return rendered

    def _list(self, node: Tag, depth: int) -> str:
        lines: list[str] = []
        ordered = node.name.lower() == "ol"
        index = int(node.get("start", 1)) if str(node.get("start", "1")).isdigit() else 1
        for item in node.find_all("li", recursive=False):
            parts: list[str] = []
            nested: list[Tag] = []
            for child in item.children:
                if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                    nested.append(child)
                else:
                    parts.append(self._node(child, depth + 1))
            body = re.sub(r"\s+", " ", "".join(parts)).strip()
            prefix = f"{index}." if ordered else "-"
            lines.append(f"{'  ' * depth}{prefix} {body}")
            for child in nested:
                lines.append(self._list(child, depth + 1).strip("\n"))
            index += 1
        return "\n\n" + "\n".join(lines) + "\n\n" if lines else ""

    def _table(self, table: Tag) -> str:
        rows: list[list[str]] = []
        header_index: int | None = None
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            values = []
            for cell in cells:
                rendered = self._children(cell).strip()
                rendered = re.sub(r"\s*\n\s*", "<br>", rendered)
                rendered = re.sub(r"[ \t]+", " ", rendered)
                values.append(rendered)
            if any(cell.name == "th" for cell in cells) and header_index is None:
                header_index = len(rows)
            rows.append(values)
        if not rows:
            return ""
        header_at = header_index if header_index is not None else 0
        headers = rows[header_at]
        body = rows[:header_at] + rows[header_at + 1 :]
        return markdown_table(headers, body)


def archive_header(source: dict[str, Any], entry: dict[str, Any], snapshot_id: str, extra_note: str | None = None) -> str:
    source_url = source.get("display_url", source["url"])
    raw_path = f"snapshots/{snapshot_id}/{entry['snapshot_path']}"
    lines = [
        "<!-- Archive metadata in this block is not source content. -->",
        (
            f"> **Archive metadata:** Source: [{source_url}]({source_url}) | "
            f"Retrieved: `{entry['fetched_at_utc']}` | Raw source: [`{raw_path}`]({raw_path}) | "
            f"SHA-256: `{entry['sha256']}`"
        ),
    ]
    if extra_note:
        lines.append(f"> **Archive note:** {extra_note}")
    lines.append("\n---\n")
    return "\n".join(lines)


def parse_csv_result(result: FetchResult) -> list[dict[str, str]]:
    text = decode_text(result.content, result.charset)
    return list(csv.DictReader(io.StringIO(text)))


def fmt_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def model_results_table(rows: list[dict[str, str]]) -> str:
    output = []
    for row in rows:
        model = row.get("Model", "")
        model_url = row.get("Model URL", "")
        if model_url:
            model = f"[{model}]({model_url})"
        output.append(
            [
                model,
                row.get("Institution", ""),
                row.get("Version", ""),
                row.get("Date", ""),
                row.get("Global Dice", ""),
                row.get("Global HIT Rate", ""),
                row.get("Instance Precision", ""),
                row.get("Instance Recall", ""),
                row.get("Instance F1", ""),
            ]
        )
    return markdown_table(
        ["Model", "Institution", "Version", "Date", "Global Dice", "Global HIT Rate", "Instance Precision", "Instance Recall", "Instance F1"],
        output,
    )


def per_category_markdown(
    rows: list[dict[str, str]], counts: dict[str, str], model_order: list[str]
) -> str:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    key_order: list[tuple[str, str]] = []
    for row in rows:
        key = (row.get("Model", ""), row.get("Version", ""))
        if key not in grouped:
            grouped[key] = {}
            key_order.append(key)
        grouped[key][row.get("Category", "")] = row
    order_lookup = {model: index for index, model in enumerate(model_order)}
    key_order.sort(key=lambda key: (order_lookup.get(key[0], len(order_lookup)), key[0], key[1]))

    parts = []
    for model, version in key_order:
        label = model + (f" - {version}" if version else "")
        table_rows = []
        for category in CATEGORY_ORDER:
            row = grouped[(model, version)].get(category, {})
            category_label = CATEGORY_LABELS.get(category, "")
            table_rows.append(
                [
                    f"**{category}** - {category_label}",
                    counts.get(category, "0"),
                    fmt_metric(row.get("Dice", 0)),
                    fmt_metric(row.get("Hit Rate", 0)),
                ]
            )
        parts.append(
            "\n".join(
                [
                    "<details>",
                    f"<summary>{html_lib.escape(label)}</summary>",
                    "",
                    markdown_table(["Category", "n", "Dice", "Hit Rate"], table_rows),
                    "",
                    "</details>",
                ]
            )
        )
    return "\n\n".join(parts)


def decode_firestore_value(value: dict[str, Any]) -> Any:
    if "nullValue" in value:
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "booleanValue" in value:
        return bool(value["booleanValue"])
    if "timestampValue" in value:
        return value["timestampValue"]
    if "mapValue" in value:
        fields = value["mapValue"].get("fields", {})
        return {key: decode_firestore_value(item) for key, item in fields.items()}
    if "arrayValue" in value:
        return [decode_firestore_value(item) for item in value["arrayValue"].get("values", [])]
    if "referenceValue" in value:
        return value["referenceValue"]
    if "geoPointValue" in value:
        return value["geoPointValue"]
    return value


def decode_firestore_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for document in payload.get("documents", []):
        row = {key: decode_firestore_value(value) for key, value in document.get("fields", {}).items()}
        row["_document_name"] = document.get("name")
        row["_create_time"] = document.get("createTime")
        row["_update_time"] = document.get("updateTime")
        output.append(row)
    return output


def rank_challenge_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluating = [row for row in rows if row.get("status") == "evaluating"]
    scored = [
        row.copy()
        for row in rows
        if row.get("status") != "evaluating" and isinstance(row.get("dice"), (int, float)) and not isinstance(row.get("dice"), bool)
    ]
    scored.sort(key=lambda row: float(row.get("dice", 0)), reverse=True)
    seen: set[str] = set()
    rank = 0
    for row in scored:
        key = str(row.get("submitterKey") or row.get("submitter") or row.get("team") or "").lower()
        if key not in seen:
            seen.add(key)
            rank += 1
            row["_archive_rank"] = rank
            row["_archive_secondary"] = False
        else:
            row["_archive_rank"] = None
            row["_archive_secondary"] = True
    return scored, evaluating


def challenge_leaderboard_table(rows: list[dict[str, Any]]) -> str:
    scored, evaluating = rank_challenge_rows(rows)
    table_rows = []
    for row in scored:
        table_rows.append(
            [
                row.get("_archive_rank") or "\u2014",
                row.get("team", ""),
                row.get("registeredTeam") or "\u2014",
                row.get("submitter") or "\u2014",
                row.get("institution") or "\u2014",
                fmt_metric(row.get("dice")),
                fmt_metric(row.get("hitRate")),
                fmt_metric(row.get("instanceF1")),
            ]
        )
    for row in evaluating:
        table_rows.append(
            [
                "\u2022",
                row.get("team", ""),
                row.get("registeredTeam") or "\u2014",
                row.get("submitter") or "\u2014",
                row.get("institution") or "\u2014",
                "Evaluating...",
                "Evaluating...",
                "Evaluating...",
            ]
        )
    if not table_rows:
        return "No results yet - submissions are evaluated periodically during the development phase."
    return markdown_table(["#", "Submission", "Team", "Submitter", "Institution", "Dice", "Hit Rate", "Instance F1"], table_rows)


def challenge_category_details(rows: list[dict[str, Any]]) -> str:
    scored, _ = rank_challenge_rows(rows)
    parts = []
    for row in scored:
        per_category = row.get("perCategory")
        if not isinstance(per_category, dict):
            continue
        order = CATEGORY_ORDER + sorted(key for key in per_category if key not in CATEGORY_ORDER)
        details = []
        for category in order:
            item = per_category.get(category)
            if not isinstance(item, dict):
                continue
            label = CATEGORY_LABELS.get(category, "")
            details.append(
                [
                    f"**{category}**" + (f" - {label}" if label else ""),
                    item.get("n", 0),
                    fmt_metric(item.get("dice")),
                    fmt_metric(item.get("hit")),
                ]
            )
        summary = str(row.get("team") or "(submission)")
        if row.get("submitter"):
            summary += f" - {row['submitter']}"
        parts.append(
            "\n".join(
                [
                    "<details>",
                    f"<summary>{html_lib.escape(summary)}</summary>",
                    "",
                    markdown_table(["Category", "n", "Dice", "Hit Rate"], details),
                    "",
                    "</details>",
                ]
            )
        )
    return "\n\n".join(parts) if parts else "No per-category results were present in the public leaderboard payload."


def parse_firebase_config(challenge_html: bytes) -> tuple[str, str]:
    text = decode_text(challenge_html)
    project_match = re.search(r"projectId\s*:\s*[\"']([^\"']+)", text)
    key_match = re.search(r"apiKey\s*:\s*[\"']([^\"']+)", text)
    if not project_match or not key_match:
        raise ValueError("Could not locate public Firebase projectId/apiKey in challenge.html")
    return project_match.group(1), key_match.group(1)


def fetch_public_leaderboard(challenge_html: bytes, fetcher: Callable[[str], FetchResult] = fetch_url) -> FetchResult:
    project_id, api_key = parse_firebase_config(challenge_html)
    base_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/leaderboard"
    url = base_url + "?" + urlencode({"pageSize": 1000, "key": api_key})
    result = fetcher(url)
    payload = json.loads(decode_text(result.content, result.charset))
    if "error" in payload:
        raise ValueError(f"Public leaderboard returned an error: {payload['error']}")
    if payload.get("nextPageToken"):
        raise ValueError("Public leaderboard exceeded 1000 documents; pagination support is required before archiving")
    if "documents" in payload and not isinstance(payload["documents"], list):
        raise ValueError("Public leaderboard documents field is not a list")
    return result


def find_section_markdown(content: bytes, base_url: str, heading_text: str) -> str:
    soup = BeautifulSoup(decode_text(content), "html.parser")
    for node in soup.find_all(["script", "style", "noscript", "template", "svg", "canvas"]):
        node.decompose()
    heading = next(
        (
            candidate
            for candidate in soup.find_all(re.compile(r"^h[1-6]$"))
            if heading_text.casefold() in candidate.get_text(" ", strip=True).casefold()
        ),
        None,
    )
    if heading is None:
        raise ValueError(f"Could not find section heading: {heading_text}")
    level = int(heading.name[1])
    anchor: Tag = heading

    def has_section_content_after(candidate: Tag) -> bool:
        for sibling in candidate.next_siblings:
            if isinstance(sibling, NavigableString) and sibling.strip():
                return True
            if not isinstance(sibling, Tag):
                continue
            sibling_heading = sibling if re.match(r"^h[1-6]$", sibling.name or "") else sibling.find(re.compile(r"^h[1-6]$"))
            if sibling_heading and int(sibling_heading.name[1]) <= level:
                return False
            if sibling.get_text(" ", strip=True) or sibling.name in {"table", "ul", "ol"}:
                return True
        return False

    while anchor.parent and isinstance(anchor.parent, Tag) and anchor.parent.name not in {"body", "html"}:
        if has_section_content_after(anchor):
            break
        anchor = anchor.parent

    fragment = BeautifulSoup("<div></div>", "html.parser").div
    assert fragment is not None
    fragment.append(BeautifulSoup(str(heading), "html.parser"))
    for sibling in anchor.next_siblings:
        sibling_heading = None
        if isinstance(sibling, Tag):
            sibling_heading = sibling if re.match(r"^h[1-6]$", sibling.name or "") else sibling.find(re.compile(r"^h[1-6]$"))
        if sibling_heading and int(sibling_heading.name[1]) <= level:
            break
        fragment.append(BeautifulSoup(str(sibling), "html.parser"))
    return MarkdownConverter(base_url).convert(str(fragment).encode("utf-8"))


def miccai_listing_markdown(content: bytes, base_url: str) -> str:
    soup = BeautifulSoup(decode_text(content), "html.parser")
    target_row = next((row for row in soup.find_all("tr") if "ReXGrounding" in row.get_text(" ", strip=True)), None)
    if target_row is None:
        raise ValueError("MICCAI challenge table does not contain ReXGrounding")
    table = target_row.find_parent("table")
    if table is None:
        raise ValueError("MICCAI ReXGrounding row is not inside a table")
    header_row = next((row for row in table.find_all("tr") if row.find("th")), None)
    selected_rows = [row for row in [header_row, target_row] if row is not None]
    columns = []
    for row in selected_rows:
        columns.append([cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)])
    if len(columns) < 2:
        raise ValueError("MICCAI challenge listing table could not be parsed")
    title = next((h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"]) if "CHALLENGE" in h.get_text(" ", strip=True).upper()), "MICCAI 2026 Challenges")
    return clean_markdown(f"# {title}\n\n{markdown_table(columns[0], [columns[1]])}\n")


def arxiv_record_markdown(content: bytes, base_url: str) -> str:
    soup = BeautifulSoup(decode_text(content), "html.parser")
    selector = "main" if soup.select_one("main") else ("#content" if soup.select_one("#content") else None)
    return MarkdownConverter(base_url).convert(content, selector)


def render_source_document(
    source: dict[str, Any],
    result: FetchResult,
    entry: dict[str, Any],
    snapshot_id: str,
    model_rows: list[dict[str, str]],
    per_category_rows: list[dict[str, str]],
    category_counts: dict[str, str],
    challenge_rows: list[dict[str, Any]],
) -> str:
    handler = source["handler"]
    note = None
    if source["id"] == "rexgroundingct_challenge":
        note = "Interactive controls are represented as text; the public runtime leaderboard is transcribed below from its archived JSON payload."
    elif source["kind"] == "html":
        note = "Interactive controls do not function in Markdown; the raw HTML remains authoritative for page behavior."
    header = archive_header(source, entry, snapshot_id, note)

    if handler == "native_markdown":
        body = decode_text(result.content, result.charset).strip() + "\n"
    elif handler == "miccai_listing":
        body = miccai_listing_markdown(result.content, source["url"])
    elif handler == "arxiv_record":
        body = arxiv_record_markdown(result.content, source["url"])
    else:
        body = MarkdownConverter(source["url"]).convert(result.content)

    if handler == "rexgrounding_index":
        models = [row.get("Model", "") for row in model_rows]
        body += (
            "\n## Archived Source Data\n\n"
            "> **Archive note:** The page's per-category selector is populated by embedded JavaScript. "
            "The complete official CSV-backed values are expanded below; displayed metrics use the page's three-decimal formatting.\n\n"
            "### Model Performance CSV\n\n"
            + model_results_table(model_rows)
            + "\n\n### Per-Category Results\n\n"
            + per_category_markdown(per_category_rows, category_counts, models)
            + "\n"
        )
    elif handler == "challenge":
        body += (
            "\n## Archived Live Leaderboard\n\n"
            "> **Archive note:** This table was decoded from the official page's intentionally public, read-only "
            "Firestore `leaderboard` collection at the snapshot time. Ranking follows the page's JavaScript exactly.\n\n"
            + challenge_leaderboard_table(challenge_rows)
            + "\n\n### Per-Submission Category Results\n\n"
            + challenge_category_details(challenge_rows)
            + "\n"
        )
    return clean_markdown(header + body)


def build_summary(
    snapshot_id: str,
    extracted_utc: str,
    results: dict[str, FetchResult],
    model_rows: list[dict[str, str]],
    challenge_rows: list[dict[str, Any]],
) -> str:
    challenge = results["rexgroundingct_challenge"]
    index = results["rexgroundingct"]
    submission = results["rexrankct_submission_guideline"]
    sections = {
        "about": find_section_markdown(index.content, "https://rexrank.ai/ReXGroundingCT/", "About ReXGroundingCT"),
        "task": find_section_markdown(challenge.content, "https://rexrank.ai/ReXGroundingCT/challenge.html", "Task"),
        "tracks": find_section_markdown(challenge.content, "https://rexrank.ai/ReXGroundingCT/challenge.html", "Challenge Tracks"),
        "dataset": find_section_markdown(challenge.content, "https://rexrank.ai/ReXGroundingCT/challenge.html", "Dataset"),
        "timeline": find_section_markdown(challenge.content, "https://rexrank.ai/ReXGroundingCT/challenge.html", "Timeline"),
        "metrics": find_section_markdown(challenge.content, "https://rexrank.ai/ReXGroundingCT/challenge.html", "Evaluation Metrics"),
    }
    guideline_text = BeautifulSoup(
        decode_text(submission.content, submission.charset),
        "html.parser",
    ).get_text(" ", strip=True)
    if "Compose an email" not in guideline_text or "Send the email" not in guideline_text:
        raise ValueError("Submission guideline no longer contains the archived email-submission wording")
    challenge_text = BeautifulSoup(
        decode_text(challenge.content, challenge.charset),
        "html.parser",
    ).get_text(" ", strip=True)
    if "This is the only submission channel" not in challenge_text:
        raise ValueError("Challenge page no longer identifies its current submission channel")

    summary = f"""# ReXGroundingCT Current Summary

<!-- This digest is archive-generated. Follow links to literal transcriptions and raw evidence. -->
> Snapshot: [`snapshots/{snapshot_id}/`](snapshots/{snapshot_id}/) | Extracted: `{extracted_utc}`

## Important Source Differences

- **Submission route:** [`rexgroundingct_challenge.md`](rexgroundingct_challenge.md) says its authenticated Register & Submit form is the only submission channel for the MICCAI challenge. [`rexrankct_submission_guideline.md`](rexrankct_submission_guideline.md) separately retains email-submission instructions. Both are transcribed unchanged; use the challenge page for the current MICCAI submission channel and the guideline as its linked prediction-format reference.
- **Dataset splits:** The base ReXrank page describes a 100-scan hosted test set. The MICCAI challenge page describes separate 200-scan validation and 300-scan test splits. The official Hugging Face update record says the MICCAI split was added separately while the base `dataset.json` remained unchanged.
- **Two leaderboards:** The main ReXGroundingCT page contains the established model benchmark. The MICCAI challenge page loads a separate live public-split leaderboard from Firestore.

## Dataset Overview

{sections['about'].strip()}

## Challenge Task

{sections['task'].strip()}

## Challenge Tracks

{sections['tracks'].strip()}

## MICCAI Dataset Splits

{sections['dataset'].strip()}

## Timeline

{sections['timeline'].strip()}

## Evaluation Metrics

{sections['metrics'].strip()}

## Main ReXGroundingCT Model Benchmark

{model_results_table(model_rows)}

## MICCAI Public-Split Leaderboard

{challenge_leaderboard_table(challenge_rows)}

## Official Companion Sources

- [Hugging Face dataset card](huggingface_dataset_card.md)
- [Dataset-generation repository](github_dataset_generation.md)
- [Paper record](arxiv_paper_record.md)
- [MICCAI 2026 challenge listing](miccai_2026_challenge_listing.md)
"""
    return clean_markdown(summary)


def build_readme(
    registry: dict[str, Any],
    snapshot_id: str,
    extracted_utc: str,
    extracted_local: str,
    entries: list[dict[str, Any]],
    warnings: list[str],
    gh_pages_commit: str | None,
) -> str:
    entry_map = {entry["id"]: entry for entry in entries}
    source_rows = []
    for source in registry["sources"]:
        entry = entry_map[source["id"]]
        raw = f"snapshots/{snapshot_id}/{entry['snapshot_path']}"
        source_rows.append(
            [
                source["title"],
                f"[{source.get('display_url', source['url'])}]({source.get('display_url', source['url'])})",
                f"[{source['output']}]({source['output']})",
                f"[`raw`]({raw})",
                f"`{entry['sha256'][:12]}...`",
            ]
        )
    data_rows = []
    for source in registry["data_sources"]:
        entry = entry_map.get(source["id"])
        if entry:
            raw = f"snapshots/{snapshot_id}/{entry['snapshot_path']}"
            data_rows.append([source["id"], f"[{source['kind']}]({raw})", entry["bytes"], f"`{entry['sha256']}`"])
        else:
            data_rows.append([source["id"], "Unavailable (optional)", "--", "--"])
    dynamic = entry_map["challenge_public_leaderboard"]
    dynamic_raw = f"snapshots/{snapshot_id}/{dynamic['snapshot_path']}"
    data_rows.append(["challenge_public_leaderboard", f"[json]({dynamic_raw})", dynamic["bytes"], f"`{dynamic['sha256']}`"])

    warnings_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None."
    commit_text = f"`{gh_pages_commit}`" if gh_pages_commit else "Unavailable; see update warnings."
    readme = f"""# Challenge Information Archive

This folder contains source-faithful Markdown transcriptions and immutable evidence snapshots for ReXGroundingCT. Raw responses are authoritative whenever an interactive website cannot be represented fully in Markdown.

## Latest Extraction

- UTC: `{extracted_utc}`
- Local: `{extracted_local}`
- Snapshot: [`snapshots/{snapshot_id}/`](snapshots/{snapshot_id}/)
- Manifest: [`snapshots/{snapshot_id}/manifest.json`](snapshots/{snapshot_id}/manifest.json)
- ReXrank `gh-pages` commit observed during extraction: {commit_text}

## Current Documents

{markdown_table(["Record", "Official source", "Literal Markdown", "Raw evidence", "SHA-256"], source_rows)}

[`SUMMARY.md`](SUMMARY.md) is a concise, source-linked digest. It is deliberately separate from the literal transcriptions above.

## Supporting Data

{markdown_table(["Artifact", "Snapshot", "Bytes", "SHA-256"], data_rows)}

The static ReXGroundingCT page embeds leaderboard values and is also supported by the archived CSV files. The MICCAI challenge leaderboard is runtime-loaded, so its public Firestore response is stored separately as JSON.

## Known Source Differences

- The challenge page states that its authenticated Register & Submit form is the only MICCAI submission channel. The linked guideline still contains instructions to submit by email. Both texts are preserved unchanged; the challenge page explicitly describes the guideline as a prediction-format reference rather than a separate submission route.
- The main ReXrank page describes a base 100-scan test set. The MICCAI challenge describes 200 validation and 300 test scans. The Hugging Face dataset card records the MICCAI split as an addition while the base dataset remains unchanged.
- The established model benchmark and the MICCAI public-split leaderboard are different leaderboards and remain separate in this archive.

## Gated Resources

The challenge links to resources inside the gated Hugging Face dataset. This updater records the official links but does not authenticate, bypass access controls, or copy unavailable content:

- [`rexrank_eval.py`](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/rexrank_eval.py)
- [`MICCAI_challenge_dataset.json`](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/MICCAI_challenge_dataset.json)

## Updating

Run from the repository root:

```powershell
python challenge_info/update_challenge_info.py
python challenge_info/update_challenge_info.py --verify-latest
```

Each successful update creates a new UTC-timestamped snapshot and refreshes the top-level Markdown files. Downloads, parsing, and validation finish before current files are promoted. A required-source failure exits nonzero and leaves the current documents unchanged.

Refresh before making submission decisions and regularly while the challenge leaderboard is active. Add future static official sources to [`sources.json`](sources.json). Dynamic sources require explicit code so a failed runtime request cannot be mistaken for an empty result.

## Integrity Policy

- Raw response bytes are stored before conversion and hashed with SHA-256.
- HTTP status, final URL, content type, ETag, Last-Modified, retrieval time, size, and hash are recorded in the manifest when available.
- Source wording, spelling, links, and displayed numeric precision are retained. Archive-generated notes are labeled explicitly.
- Only the intentionally public Firestore `leaderboard` collection is read. Private teams, submissions, authentication data, and withheld scores are outside scope.
- Full dataset binaries and paper PDFs are outside scope.

## Update Warnings

{warnings_text}
"""
    return clean_markdown(readme)


def generated_manifest_entry(path: Path, snapshot_root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(snapshot_root).as_posix(),
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def verify_snapshot(snapshot: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        return [f"Missing manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Invalid manifest: {exc}"]
    for entry in [*manifest.get("sources", []), *manifest.get("generated", [])]:
        relative = entry.get("snapshot_path") or entry.get("path")
        if not relative:
            errors.append(f"Manifest entry has no path: {entry.get('id', '<generated>')}")
            continue
        path = snapshot / relative
        if not path.is_file():
            errors.append(f"Missing archived file: {relative}")
            continue
        content = path.read_bytes()
        actual_hash = sha256_bytes(content)
        if actual_hash != entry.get("sha256"):
            errors.append(f"SHA-256 mismatch: {relative}")
        if len(content) != entry.get("bytes"):
            errors.append(f"Byte-count mismatch: {relative}")
    return errors


def latest_snapshot(root: Path = ROOT) -> Path:
    snapshots = root / "snapshots"
    candidates = sorted(path for path in snapshots.glob("20*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError("No completed snapshots were found")
    return candidates[-1]


def promote_documents(generated_dir: Path, root: Path, output_names: list[str]) -> None:
    promotion_dir = Path(tempfile.mkdtemp(prefix=".archive-promotion-", dir=root))
    backup_dir = promotion_dir / "backup"
    staged_dir = promotion_dir / "staged"
    backup_dir.mkdir()
    staged_dir.mkdir()
    existing: set[str] = set()
    try:
        for name in output_names:
            source = generated_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Generated document is missing: {name}")
            shutil.copy2(source, staged_dir / name)
            target = root / name
            if target.exists():
                existing.add(name)
                shutil.copy2(target, backup_dir / name)
        normalize_windows_acl(promotion_dir)
        promoted: list[str] = []
        try:
            for name in output_names:
                os.replace(staged_dir / name, root / name)
                promoted.append(name)
        except Exception:
            for name in promoted:
                target = root / name
                if name in existing:
                    os.replace(backup_dir / name, target)
                elif target.exists():
                    target.unlink()
            raise
    finally:
        shutil.rmtree(promotion_dir, ignore_errors=True)


def run_update(root: Path = ROOT, fetcher: Callable[[str], FetchResult] = fetch_url) -> Path:
    registry = load_registry(root)
    now = utc_now()
    snapshot_id = snapshot_id_for(now)
    extracted_utc = iso_utc(now)
    extracted_local = datetime.now().astimezone().isoformat(timespec="seconds")
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    final_snapshot = snapshots_dir / snapshot_id
    suffix = 1
    while final_snapshot.exists():
        final_snapshot = snapshots_dir / f"{snapshot_id}-{suffix:02d}"
        suffix += 1
    snapshot_id = final_snapshot.name
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=snapshots_dir))
    entries: list[dict[str, Any]] = []
    results: dict[str, FetchResult] = {}
    warnings: list[str] = []
    gh_pages_commit: str | None = None

    try:
        for source in [*registry["sources"], *registry["data_sources"]]:
            try:
                result = fetcher(source["url"])
                validate_fetch(source, result)
            except Exception as exc:
                if source.get("required", True):
                    raise RuntimeError(f"Required source failed: {source['id']}: {exc}") from exc
                warnings.append(f"Optional source `{source['id']}` was unavailable: {exc}")
                continue
            results[source["id"]] = result
            write_bytes(staging / source["snapshot_path"], result.content)
            entries.append(manifest_entry(source, result))

        challenge_result = results["rexgroundingct_challenge"]
        leaderboard_result = fetch_public_leaderboard(challenge_result.content, fetcher)
        leaderboard_payload = json.loads(decode_text(leaderboard_result.content, leaderboard_result.charset))
        challenge_rows = decode_firestore_documents(leaderboard_payload)
        leaderboard_source = {
            "id": "challenge_public_leaderboard",
            "category": "dynamic_data",
            "required": True,
            "snapshot_path": "raw/firestore.googleapis.com/rexgrounding-challenge/leaderboard.json",
        }
        write_bytes(staging / leaderboard_source["snapshot_path"], leaderboard_result.content)
        entries.append(manifest_entry(leaderboard_source, leaderboard_result))

        model_rows = parse_csv_result(results["rexgroundingct_model_results"])
        per_category_rows = parse_csv_result(results["rexgroundingct_per_category_results"])
        count_rows = parse_csv_result(results["rexgroundingct_category_counts"])
        category_counts = {row["Category"]: row["n"] for row in count_rows}
        if not model_rows or not per_category_rows or set(category_counts) != set(CATEGORY_ORDER):
            raise ValueError("Official CSV files did not contain the expected leaderboard/category records")

        branch_result = results.get("rexrank_gh_pages_branch")
        if branch_result:
            try:
                gh_pages_commit = json.loads(decode_text(branch_result.content, branch_result.charset))["commit"]["sha"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                warnings.append(f"Could not decode the optional gh-pages commit metadata: {exc}")

        entry_map = {entry["id"]: entry for entry in entries}
        documents: dict[str, str] = {}
        for source in registry["sources"]:
            documents[source["output"]] = render_source_document(
                source,
                results[source["id"]],
                entry_map[source["id"]],
                snapshot_id,
                model_rows,
                per_category_rows,
                category_counts,
                challenge_rows,
            )
        documents["SUMMARY.md"] = build_summary(snapshot_id, extracted_utc, results, model_rows, challenge_rows)
        documents["README.md"] = build_readme(
            registry,
            snapshot_id,
            extracted_utc,
            extracted_local,
            entries,
            warnings,
            gh_pages_commit,
        )

        required_output_text = {
            "rexgroundingct.md": ["ReXGroundingCT", "Archived Source Data", "Per-Category Results"],
            "rexgroundingct_challenge.md": ["ReXGrounding Challenge", "Archived Live Leaderboard", "Evaluation Metrics"],
            "rexrankct_submission_guideline.md": ["ReXrankCT Submission Instructions", "How to Submit"],
            "SUMMARY.md": ["Important Source Differences", "MICCAI Public-Split Leaderboard"],
            "README.md": ["Latest Extraction", "Integrity Policy"],
        }
        for name, needles in required_output_text.items():
            missing = [needle for needle in needles if needle not in documents[name]]
            if missing:
                raise ValueError(f"Generated {name} is missing required text: {missing}")

        generated_dir = staging / "generated"
        for name, content in documents.items():
            write_text(generated_dir / name, content)
        generated_entries = [generated_manifest_entry(generated_dir / name, staging) for name in sorted(documents)]
        manifest = {
            "schema_version": 1,
            "tool_version": TOOL_VERSION,
            "snapshot_id": snapshot_id,
            "extracted_at_utc": extracted_utc,
            "extracted_at_local": extracted_local,
            "rexrank_gh_pages_commit": gh_pages_commit,
            "sources": entries,
            "generated": generated_entries,
            "warnings": warnings,
            "privacy_scope": "Public web sources and the public read-only Firestore leaderboard collection only.",
        }
        write_text(staging / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

        errors = verify_snapshot(staging)
        if errors:
            raise ValueError("Snapshot verification failed: " + "; ".join(errors))

        normalize_windows_acl(staging)
        os.replace(staging, final_snapshot)
        promote_documents(final_snapshot / "generated", root, sorted(documents))
        return final_snapshot
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_latest(root: Path = ROOT) -> Path:
    snapshot = latest_snapshot(root)
    errors = verify_snapshot(snapshot)
    if errors:
        raise ValueError("\n".join(errors))
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest.get("generated", []):
        name = Path(entry["path"]).name
        current = root / name
        archived = snapshot / entry["path"]
        if not current.is_file():
            raise ValueError(f"Current generated document is missing: {name}")
        if current.read_bytes() != archived.read_bytes():
            raise ValueError(f"Current generated document does not match latest snapshot: {name}")
    return snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-latest",
        action="store_true",
        help="Verify hashes, byte counts, and current generated files without using the network.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_latest:
            snapshot = verify_latest(ROOT)
            print(f"Verified snapshot: {snapshot}")
        else:
            snapshot = run_update(ROOT)
            print(f"Created snapshot: {snapshot}")
            print("Refreshed current Markdown documents in challenge_info/.")
        return 0
    except Exception as exc:
        print(f"Archive update failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
