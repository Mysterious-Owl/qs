#!/usr/bin/env python3
"""Local reader for the interview-prep docs.

Usage:
    python reader.py                # serve on http://localhost:8000 and open a browser
    python reader.py --port 9000    # pick a port
    python reader.py --no-browser   # don't auto-open

Renders every .md and .ipynb in this folder with a sidebar, per-document table of
contents, full-text search (Ctrl+K), and light/dark themes. Internal links between
docs are rewritten so navigation works inside the reader.
"""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    import markdown
except ImportError:
    sys.exit("This reader needs the 'markdown' package:\n    pip install markdown")

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}

# Static assets a document may reference (figures, diagrams).
ASSET_TYPES = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".pdf": "application/pdf",
}

# Source files a document may link to — served as readable plain text.
TEXT_TYPES = {".py", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".sql"}

# Sidebar group order. FOLDER_ORDER comes first in the order given, FOLDER_LAST
# comes last in the order given, and anything unlisted sorts alphabetically in
# between — so adding a folder never silently lands it at the top or bottom.
FOLDER_ORDER = [
    "",                  # root README -> "Home"
    "ML_Fundamentals",
    "NLP",
    "Question_Bank",
    "AI_Engineer",
    "ML_Engineer",
    "Data_Scientist",
    "AI_Researcher",
    "AI_Architect",
]
FOLDER_LAST = ["Google", "Example_Company"]


def folder_rank(folder: str) -> tuple:
    """Sort key for a sidebar group. Shared by the server and the static build."""
    if folder in FOLDER_ORDER:
        return (0, FOLDER_ORDER.index(folder), "")
    if folder in FOLDER_LAST:
        return (2, FOLDER_LAST.index(folder), "")
    return (1, 0, folder)

MD_EXTENSIONS = ["tables", "fenced_code", "toc", "sane_lists", "attr_list", "md_in_html"]

# Renders mutate shared Doc objects; the server is threaded, so serialise them.
RENDER_LOCK = threading.Lock()


# --------------------------------------------------------------------------- docs

class Doc:
    """One rendered document."""

    def __init__(self, path: Path):
        self.path = path
        self.rel = path.relative_to(ROOT).as_posix()
        self.folder = posixpath.dirname(self.rel)
        self.is_notebook = path.suffix == ".ipynb"
        self.body = ""
        self.toc: list[dict] = []
        self.title = path.stem
        self.plain = ""
        self.mtime = -1.0  # source mtime the current render came from

    @property
    def url(self) -> str:
        return "/doc/" + self.rel

    @property
    def label(self) -> str:
        """Sidebar label: strip the numeric prefix but keep it for ordering."""
        name = self.path.stem
        name = re.sub(r"^\d+[_\-]", "", name)
        return name.replace("_", " ")


def discover() -> list[Doc]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in (".md", ".ipynb"):
            files.append(path)

    def sort_key(p: Path):
        rel = p.relative_to(ROOT).as_posix()
        folder = posixpath.dirname(rel)
        # README first within a folder, then numeric prefix, then name.
        is_readme = 0 if p.stem.lower() == "readme" else 1
        num_match = re.match(r"^(\d+)", p.stem)
        num = int(num_match.group(1)) if num_match else 9999
        return (folder_rank(folder), is_readme, num, p.stem.lower())

    return [Doc(p) for p in sorted(files, key=sort_key)]


# ----------------------------------------------------------------------- render

def make_converter() -> markdown.Markdown:
    return markdown.Markdown(extensions=MD_EXTENSIONS, output_format="html5")


def notebook_to_markdown(path: Path) -> str:
    """Flatten a notebook into markdown: md cells as-is, code cells as fenced blocks."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"> Could not read notebook: {exc}"

    lang = (
        nb.get("metadata", {}).get("language_info", {}).get("name")
        or nb.get("metadata", {}).get("kernelspec", {}).get("language")
        or "python"
    )

    parts: list[str] = []
    code_n = 0
    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        if not text.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(text)
        elif cell.get("cell_type") == "code":
            code_n += 1
            parts.append(f"###### Cell {code_n}\n\n```{lang}\n{text.rstrip()}\n```")
    return "\n\n".join(parts)


def rewrite_links(body: str, doc: Doc) -> str:
    """Point relative links at .md/.ipynb files to reader routes; mark external links."""

    def fix(match: re.Match) -> str:
        quote, href = match.group(1), match.group(2)
        parsed = urlparse(href)
        if parsed.scheme or href.startswith(("#", "/", "mailto:")):
            if parsed.scheme in ("http", "https"):
                return f'target="_blank" rel="noopener" href={quote}{href}{quote}'
            return match.group(0)

        target, _, fragment = href.partition("#")
        if not target:
            return match.group(0)

        resolved = posixpath.normpath(posixpath.join(doc.folder, target))
        candidate = ROOT / resolved
        if candidate.is_dir():
            # A folder link — aim at its README if there is one.
            readme = candidate / "README.md"
            if readme.exists():
                resolved = readme.relative_to(ROOT).as_posix()
            else:
                return f'href={quote}#{quote} class="dead-link"'
        elif not candidate.exists():
            return f'href={quote}#{quote} class="dead-link"'

        suffix = "#" + fragment if fragment else ""
        return f'href={quote}/doc/{resolved}{suffix}{quote}'

    return re.sub(r'href=(["\'])([^"\']+)\1', fix, body)


def strip_tags(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def render(doc: Doc, force: bool = False) -> None:
    """Populate doc.body / doc.toc / doc.title / doc.plain.

    Skips the work when the file hasn't changed since the last render, so page
    loads are instant but editing a file and refreshing still shows the update.
    """
    try:
        mtime = doc.path.stat().st_mtime
    except OSError:
        mtime = -1.0

    with RENDER_LOCK:
        if not force and doc.body and mtime == doc.mtime:
            return
        _render_locked(doc, mtime)


def _render_locked(doc: Doc, mtime: float) -> None:
    doc.mtime = mtime

    if doc.is_notebook:
        raw = notebook_to_markdown(doc.path)
    else:
        try:
            raw = doc.path.read_text(encoding="utf-8")
        except OSError as exc:
            raw = f"> Could not read file: {exc}"

    converter = make_converter()
    body = converter.convert(raw)

    # First H1 becomes the document title.
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    doc.title = strip_tags(h1.group(1)) if h1 else doc.label

    # Flatten the nested TOC, keeping each heading's real level (h1=1, h2=2, ...).
    flat: list[dict] = []

    def walk(items):
        for item in items:
            level = item.get("level", 1)
            if level <= 3:
                flat.append({"id": item["id"], "name": strip_tags(item["name"]), "level": level})
            walk(item.get("children", []))

    walk(getattr(converter, "toc_tokens", []))

    # Drop the leading H1 — it already appears as the page heading.
    if flat and flat[0]["level"] == 1:
        flat = flat[1:]

    # Indent relative to the shallowest remaining heading, so a doc built from
    # h2/h3 and one built from h1/h3 both get a sensible two-level TOC.
    if flat:
        base = min(e["level"] for e in flat)
        for e in flat:
            e["indent"] = min(e["level"] - base, 1)
    doc.toc = flat[:80]

    doc.body = rewrite_links(body, doc)
    doc.plain = strip_tags(body)


# -------------------------------------------------------------------------- page

CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg: #fbfbfa;
  --surface: #ffffff;
  --surface-2: #f4f4f2;
  --border: #e3e3df;
  --text: #23221f;
  --text-dim: #6b6a65;
  --text-faint: #9a9995;
  --accent: #1d6fd0;
  --accent-soft: #e8f1fc;
  --code-bg: #f5f5f3;
  --code-text: #2d2b28;
  --mark: #fff3b8;
  --sidebar-w: 290px;
  --toc-w: 224px;
  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}

:root[data-theme="dark"] {
  --bg: #17171a;
  --surface: #1e1e22;
  --surface-2: #26262b;
  --border: #33333a;
  --text: #e6e5e2;
  --text-dim: #a3a29d;
  --text-faint: #74736e;
  --accent: #6ea8f0;
  --accent-soft: #1c2a3d;
  --code-bg: #24242a;
  --code-text: #dcdad5;
  --mark: #5c4d16;
}

html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
a.dead-link { color: var(--text-faint); cursor: not-allowed; text-decoration: line-through; }

/* ---------- layout ---------- */
.layout { display: flex; min-height: 100vh; }

.sidebar {
  width: var(--sidebar-w);
  flex: 0 0 var(--sidebar-w);
  background: var(--surface);
  border-right: 1px solid var(--border);
  height: 100vh;
  position: sticky;
  top: 0;
  display: flex;
  flex-direction: column;
}

.brand {
  padding: 18px 20px 14px;
  border-bottom: 1px solid var(--border);
}
.brand h1 { margin: 0; font-size: 14px; letter-spacing: .02em; font-weight: 650; }
.brand p { margin: 3px 0 0; font-size: 12px; color: var(--text-dim); }

.search-btn {
  margin: 12px 14px;
  padding: 7px 11px;
  display: flex; align-items: center; gap: 8px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 7px;
  color: var(--text-dim);
  font: inherit; font-size: 13px;
  cursor: pointer; text-align: left;
}
.search-btn:hover { border-color: var(--accent); color: var(--text); }
.search-btn kbd {
  margin-left: auto;
  font: 11px var(--mono);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px; padding: 1px 5px;
  color: var(--text-faint);
}

.nav { flex: 1; overflow-y: auto; padding: 0 8px 24px; }
.nav-group { margin-bottom: 10px; }

/* Group header doubles as the expand/collapse control. */
.nav-group > .group-name {
  display: flex; align-items: center; gap: 6px;
  width: 100%;
  font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--text-faint); font-weight: 650;
  padding: 6px 10px 5px;
  background: none; border: none; border-radius: 6px;
  font-family: inherit;
  cursor: pointer; text-align: left;
}
.nav-group > .group-name:hover { background: var(--surface-2); color: var(--text-dim); }
.nav-group > .group-name .chev {
  font-size: 10px; line-height: 1;
  transition: transform .15s ease;
}
.nav-group.collapsed > .group-name .chev { transform: rotate(-90deg); }
.nav-group > .group-name .group-label { overflow-wrap: anywhere; }
.nav-group > .group-name .group-count {
  margin-left: auto;
  font-size: 10px; font-weight: 600;
  color: var(--text-faint);
  background: var(--surface-2);
  border-radius: 9px; padding: 1px 6px;
  font-variant-numeric: tabular-nums;
}
.nav-group.collapsed > .group-items { display: none; }
.nav a {
  display: block;
  padding: 5px 12px;
  margin: 1px 0;
  border-radius: 6px;
  color: var(--text-dim);
  font-size: 13.5px;
  line-height: 1.4;
  overflow-wrap: break-word;
}
.nav a:hover { background: var(--surface-2); color: var(--text); text-decoration: none; }
.nav a.active { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.nav a .num { color: var(--text-faint); font-variant-numeric: tabular-nums; margin-right: 5px; }
.nav a.active .num { color: var(--accent); }

.sidebar-foot {
  border-top: 1px solid var(--border);
  padding: 10px 14px;
  display: flex; align-items: center; gap: 8px;
}
.icon-btn {
  background: none; border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-dim); cursor: pointer;
  font: inherit; font-size: 12px; padding: 4px 9px;
}
.icon-btn:hover { color: var(--text); border-color: var(--accent); }

/* ---------- main ---------- */
.main { flex: 1; min-width: 0; display: flex; justify-content: center; }
.column { flex: 1; min-width: 0; max-width: 1180px; display: flex; gap: 28px; padding: 0 32px; }

.article-wrap { flex: 1; min-width: 0; padding: 40px 0 120px; }
article { max-width: 76ch; }

.crumbs {
  font-size: 12px; color: var(--text-faint);
  margin-bottom: 18px; font-family: var(--mono);
}

.toc {
  width: var(--toc-w); flex: 0 0 var(--toc-w);
  position: sticky; top: 0; align-self: flex-start;
  max-height: 100vh; overflow-y: auto;
  padding: 44px 0 40px; font-size: 12.5px;
}
.toc h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--text-faint); margin: 0 0 8px; font-weight: 650;
}
.toc a {
  display: block; padding: 3px 10px;
  color: var(--text-dim); line-height: 1.4;
  border-left: 2px solid var(--border);
}
.toc a.sub { padding-left: 22px; font-size: 12px; }
.toc a:hover { color: var(--text); text-decoration: none; }
.toc a.active { color: var(--accent); border-left-color: var(--accent); font-weight: 600; }

/* ---------- prose ---------- */
article h1 {
  font-family: var(--serif);
  font-size: 2.05rem; line-height: 1.2;
  margin: 0 0 26px; font-weight: 650; letter-spacing: -.015em;
}
article h2 {
  font-family: var(--serif);
  font-size: 1.42rem; margin: 46px 0 14px;
  padding-bottom: 7px; border-bottom: 1px solid var(--border);
  font-weight: 650; letter-spacing: -.01em;
}
article h3 { font-size: 1.13rem; margin: 32px 0 10px; font-weight: 650; }
article h4, article h5 { font-size: 1rem; margin: 24px 0 8px; font-weight: 650; }
article h6 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--text-faint); margin: 26px 0 6px; font-weight: 650;
}
article h2:first-child, article h3:first-child { margin-top: 0; }

article p { margin: 0 0 15px; }
article ul, article ol { margin: 0 0 15px; padding-left: 24px; }
article li { margin: 4px 0; }
article li > ul, article li > ol { margin: 4px 0; }
article strong { font-weight: 650; }
article hr { border: none; border-top: 1px solid var(--border); margin: 36px 0; }

article blockquote {
  margin: 0 0 16px; padding: 2px 0 2px 16px;
  border-left: 3px solid var(--accent);
  color: var(--text-dim);
}
article blockquote p:last-child { margin-bottom: 0; }

article code {
  font-family: var(--mono); font-size: .875em;
  background: var(--code-bg); color: var(--code-text);
  padding: .13em .38em; border-radius: 4px;
  overflow-wrap: break-word;
}
article pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
  margin: 0 0 16px;
  line-height: 1.55;
}
article pre code {
  background: none; padding: 0; font-size: 13px;
  white-space: pre; overflow-wrap: normal;
}

/* Figures: transparent SVGs sized to the prose column, scrollable if wider. */
article img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 4px auto 20px;
}
.figure-scroll {
  overflow-x: auto;
  margin: 0 0 20px;
  padding-bottom: 4px;
}
.figure-scroll img { margin-bottom: 0; min-width: 620px; }

.table-scroll { overflow-x: auto; margin: 0 0 18px; }
article table { border-collapse: collapse; width: 100%; font-size: 14px; }
article th, article td {
  border: 1px solid var(--border);
  padding: 8px 12px; text-align: left; vertical-align: top;
}
article th { background: var(--surface-2); font-weight: 650; }
article tbody tr:nth-child(even) { background: color-mix(in srgb, var(--surface-2) 45%, transparent); }
article td code { white-space: nowrap; }

.pager {
  display: flex; gap: 12px; margin-top: 56px;
  padding-top: 22px; border-top: 1px solid var(--border);
}
.pager a {
  flex: 1; padding: 12px 15px;
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface);
}
.pager a:hover { border-color: var(--accent); text-decoration: none; }
.pager .dir { display: block; font-size: 11px; color: var(--text-faint); margin-bottom: 3px; }
.pager .name { color: var(--text); font-size: 14px; font-weight: 600; }
.pager .next { text-align: right; }

/* ---------- search ---------- */
.overlay {
  position: fixed; inset: 0; z-index: 50;
  background: rgba(0,0,0,.45);
  display: none; justify-content: center; align-items: flex-start;
  padding-top: 11vh;
}
.overlay.open { display: flex; }
.palette {
  width: min(660px, 92vw);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 18px 48px rgba(0,0,0,.28);
  overflow: hidden;
}
.palette input {
  width: 100%; border: none; outline: none;
  padding: 15px 18px; font: inherit; font-size: 15px;
  background: transparent; color: var(--text);
  border-bottom: 1px solid var(--border);
}
.results { max-height: 56vh; overflow-y: auto; padding: 6px; }
.results a { display: block; padding: 9px 12px; border-radius: 7px; }
.results a:hover, .results a.sel { background: var(--surface-2); text-decoration: none; }
.results .r-title { color: var(--text); font-size: 14px; font-weight: 600; }
.results .r-path { color: var(--text-faint); font-size: 11.5px; font-family: var(--mono); margin-top: 1px; }
.results .r-snippet { color: var(--text-dim); font-size: 12.5px; margin-top: 3px; line-height: 1.45; }
.results mark { background: var(--mark); color: inherit; border-radius: 2px; padding: 0 1px; }
.results .empty { padding: 22px; text-align: center; color: var(--text-faint); font-size: 13.5px; }

/* ---------- selection toolbar ---------- */
.sel-popup {
  position: fixed;
  z-index: 60;
  display: flex; align-items: center; gap: 2px;
  padding: 4px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  box-shadow: 0 6px 24px rgba(0,0,0,.22), 0 0 0 1px rgba(0,0,0,.03);
  animation: selPopIn .12s ease;
}
:root[data-theme="dark"] .sel-popup,
:root:not([data-theme="light"]) .sel-popup {
  box-shadow: 0 6px 26px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.05);
}
@keyframes selPopIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; } }

.sel-btn {
  display: flex; align-items: center; gap: 5px;
  min-height: 28px; padding: 5px 9px;
  background: none; border: none; border-radius: 7px;
  color: var(--text-dim);
  font: inherit; font-size: 12px; font-weight: 600;
  cursor: pointer; white-space: nowrap;
}
.sel-btn:hover { background: var(--surface-2); color: var(--text); }
.sel-btn svg { width: 13px; height: 13px; flex-shrink: 0; }
.sel-div { width: 1px; height: 18px; background: var(--border); flex-shrink: 0; margin: 0 2px; }

.toast {
  position: fixed; bottom: 26px; left: 50%;
  transform: translateX(-50%) translateY(6px);
  z-index: 70;
  padding: 8px 16px;
  background: var(--text); color: var(--bg);
  border-radius: 8px;
  font-size: 12.5px; font-weight: 600;
  opacity: 0; transition: opacity .16s ease, transform .16s ease;
  pointer-events: none;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

/* ---------- responsive ---------- */
.menu-toggle { display: none; }
@media (max-width: 1150px) { .toc { display: none; } }
@media (max-width: 860px) {
  .layout { display: block; }
  .sidebar {
    position: fixed; z-index: 40; top: 0; left: 0;
    transform: translateX(-100%); transition: transform .18s ease;
    box-shadow: 0 0 30px rgba(0,0,0,.2);
  }
  .sidebar.open { transform: none; }
  .column { padding: 0 20px; }
  .article-wrap { padding-top: 68px; }
  .menu-toggle {
    display: block; position: fixed; top: 12px; left: 12px; z-index: 45;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 11px; cursor: pointer;
    color: var(--text); font: inherit; font-size: 15px;
  }
}

@media print {
  .sidebar, .toc, .pager, .menu-toggle, .overlay { display: none !important; }
  .column { padding: 0; } article { max-width: none; }
}
"""

JS = r"""
(function () {
  // Base for all generated links. Dev server sets "/"; the static build sets
  // a relative prefix ("", "../", ...) so it works under a GitHub Pages subpath.
  var ROOT = document.body.getAttribute('data-root') || '';
  // ---- theme -------------------------------------------------------------
  var root = document.documentElement;
  function setTheme(t) {
    root.setAttribute('data-theme', t);
    try { localStorage.setItem('reader-theme', t); } catch (e) {}
    var btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = t === 'dark' ? 'Light mode' : 'Dark mode';
  }
  var stored = null;
  try { stored = localStorage.getItem('reader-theme'); } catch (e) {}
  setTheme(stored || (window.matchMedia &&
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

  var themeBtn = document.getElementById('theme-btn');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });

  // ---- wrap tables so wide ones scroll instead of blowing out the page ---
  document.querySelectorAll('article table').forEach(function (t) {
    if (t.parentElement.classList.contains('table-scroll')) return;
    var box = document.createElement('div');
    box.className = 'table-scroll';
    t.parentNode.insertBefore(box, t);
    box.appendChild(t);
  });

  // ---- sidebar groups: expand / collapse, remembered per group -----------
  var COLLAPSE_KEY = 'reader-collapsed-groups';
  var groups = Array.prototype.slice.call(document.querySelectorAll('.nav-group'));

  function readCollapsed() {
    try { return JSON.parse(localStorage.getItem(COLLAPSE_KEY) || '[]') || []; }
    catch (e) { return []; }
  }
  function writeCollapsed(list) {
    try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify(list)); } catch (e) {}
  }
  function setCollapsed(group, on) {
    group.classList.toggle('collapsed', on);
    var btn = group.querySelector('.group-name');
    if (btn) btn.setAttribute('aria-expanded', on ? 'false' : 'true');
  }

  var collapsed = readCollapsed();
  groups.forEach(function (group) {
    var name = group.getAttribute('data-group') || '';
    // The group holding the current page always opens, so you can see where you are.
    var holdsCurrent = !!group.querySelector('a.active');
    if (!holdsCurrent && collapsed.indexOf(name) !== -1) setCollapsed(group, true);

    var btn = group.querySelector('.group-name');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var nowCollapsed = !group.classList.contains('collapsed');
      setCollapsed(group, nowCollapsed);
      var at = collapsed.indexOf(name);
      if (nowCollapsed && at === -1) collapsed.push(name);
      if (!nowCollapsed && at !== -1) collapsed.splice(at, 1);
      writeCollapsed(collapsed);
    });
  });

  var allBtn = document.getElementById('toggle-all-btn');
  function refreshAllBtn() {
    if (!allBtn) return;
    var anyOpen = groups.some(function (g) { return !g.classList.contains('collapsed'); });
    allBtn.textContent = anyOpen ? 'Collapse all' : 'Expand all';
    allBtn.dataset.action = anyOpen ? 'collapse' : 'expand';
  }
  if (allBtn) {
    refreshAllBtn();
    allBtn.addEventListener('click', function () {
      var collapseThem = allBtn.dataset.action === 'collapse';
      collapsed = [];
      groups.forEach(function (g) {
        setCollapsed(g, collapseThem);
        if (collapseThem) collapsed.push(g.getAttribute('data-group') || '');
      });
      writeCollapsed(collapsed);
      refreshAllBtn();
    });
    groups.forEach(function (g) {
      var b = g.querySelector('.group-name');
      if (b) b.addEventListener('click', refreshAllBtn);
    });
  }

  // ---- wide figures scroll rather than shrink to illegibility ------------
  function fitFigure(img) {
    if (img.parentElement.classList.contains('figure-scroll')) return;
    var avail = img.parentElement.clientWidth || 700;
    if (img.naturalWidth && img.naturalWidth > avail * 1.15) {
      var box = document.createElement('div');
      box.className = 'figure-scroll';
      img.parentNode.insertBefore(box, img);
      box.appendChild(img);
    }
  }
  document.querySelectorAll('article img').forEach(function (img) {
    if (img.complete) fitFigure(img);
    else img.addEventListener('load', function () { fitFigure(img); });
  });

  // ---- sidebar (small screens) -------------------------------------------
  var sidebar = document.querySelector('.sidebar');
  var menuBtn = document.querySelector('.menu-toggle');
  if (menuBtn) menuBtn.addEventListener('click', function () { sidebar.classList.toggle('open'); });

  // ---- TOC scroll-spy ----------------------------------------------------
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  if (tocLinks.length) {
    var targets = tocLinks.map(function (a) {
      return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
    });
    var spy = function () {
      var best = -1;
      for (var i = 0; i < targets.length; i++) {
        if (targets[i] && targets[i].getBoundingClientRect().top <= 120) best = i;
      }
      tocLinks.forEach(function (a, i) { a.classList.toggle('active', i === best); });
    };
    var queued = false;
    window.addEventListener('scroll', function () {
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () { spy(); queued = false; });
    }, { passive: true });
    spy();
  }

  // ---- selection toolbar: copy / ask an assistant ------------------------
  // Appears when you select text inside the article. Positioned from the
  // pointer, preferring the side the selection was dragged toward.
  var SEL_MIN = 3;          // ignore stray clicks
  var SEL_MAX_PROMPT = 1500; // keep the assistant URL under practical limits
  var selTimer = null, selX = 0, selY = 0;

  function toast(msg) {
    var t = document.createElement('div');
    t.className = 'toast';
    t.textContent = msg;
    document.body.appendChild(t);
    requestAnimationFrame(function () { t.classList.add('show'); });
    setTimeout(function () {
      t.classList.remove('show');
      setTimeout(function () { t.remove(); }, 200);
    }, 1400);
  }

  function hideSelPopup() {
    var p = document.getElementById('selPopup');
    if (p) p.remove();
  }

  function askUrl(kind, text) {
    var q = text.length > SEL_MAX_PROMPT ? text.slice(0, SEL_MAX_PROMPT) + '…' : text;
    var prompt = encodeURIComponent('Explain this:\n\n"' + q + '"');
    return kind === 'chatgpt'
      ? 'https://chatgpt.com/?prompt=' + prompt
      : 'https://claude.ai/new?q=' + prompt;
  }

  function showSelPopup(text, cx, cy, upward) {
    hideSelPopup();
    var popup = document.createElement('div');
    popup.id = 'selPopup';
    popup.className = 'sel-popup';
    popup.innerHTML =
      '<button class="sel-btn" data-act="copy" title="Copy selection">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">' +
        '<rect x="9" y="9" width="13" height="13" rx="2"/>' +
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</button>' +
      '<div class="sel-div"></div>' +
      '<button class="sel-btn" data-act="chatgpt" title="Ask ChatGPT about this">ChatGPT</button>' +
      '<div class="sel-div"></div>' +
      '<button class="sel-btn" data-act="claude" title="Ask Claude about this">Claude</button>';
    document.body.appendChild(popup);

    var pw = popup.offsetWidth || 250, ph = popup.offsetHeight || 36, gap = 10;
    var vw = window.innerWidth, vh = window.innerHeight;
    var above = cy - gap, below = vh - cy - gap, top;
    if (upward) {
      top = above >= ph ? cy - ph - gap : (below >= ph ? cy + gap
            : (above >= below ? cy - ph - gap : cy + gap));
    } else {
      top = below >= ph ? cy + gap : (above >= ph ? cy - ph - gap
            : (below >= above ? cy + gap : cy - ph - gap));
    }
    popup.style.top = Math.max(gap, Math.min(top, vh - ph - gap)) + 'px';
    popup.style.left = Math.max(gap, Math.min(cx - pw / 2, vw - pw - gap)) + 'px';

    popup.addEventListener('click', function (e) {
      var btn = e.target.closest('.sel-btn');
      if (!btn) return;
      e.stopPropagation();
      var act = btn.dataset.act;
      if (act === 'copy') {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text)
            .then(function () { toast('Copied'); })
            .catch(function () { toast('Copy failed'); });
        } else {
          try { document.execCommand('copy'); toast('Copied'); } catch (err) { toast('Copy failed'); }
        }
      } else {
        window.open(askUrl(act, text), '_blank', 'noopener');
      }
      hideSelPopup();
    });
  }

  // Anchor position comes from the pointer, tracked continuously.
  document.addEventListener('mousemove', function (e) {
    selX = e.clientX; selY = e.clientY;
  }, { passive: true });
  document.addEventListener('touchmove', function (e) {
    if (e.touches[0]) { selX = e.touches[0].clientX; selY = e.touches[0].clientY; }
  }, { passive: true });

  function onSelEnd() {
    clearTimeout(selTimer);
    selTimer = setTimeout(function () {
      var sel = window.getSelection();
      var text = sel ? sel.toString().trim() : '';
      var article = document.querySelector('article');
      if (!text || text.length < SEL_MIN || !article) { hideSelPopup(); return; }
      try {
        var node = sel.getRangeAt(0).commonAncestorContainer;
        var el = node.nodeType === 3 ? node.parentElement : node;
        if (!article.contains(el)) { hideSelPopup(); return; }
        // Selecting backwards should put the toolbar above, not under the cursor.
        var upward = false;
        if (sel.anchorNode && sel.focusNode) {
          if (sel.anchorNode === sel.focusNode) {
            upward = sel.anchorOffset > sel.focusOffset;
          } else {
            var cmp = sel.anchorNode.compareDocumentPosition(sel.focusNode);
            upward = !(cmp & Node.DOCUMENT_POSITION_FOLLOWING);
          }
        }
        showSelPopup(text, selX, selY, upward);
      } catch (err) { hideSelPopup(); }
    }, 30);
  }

  document.addEventListener('mouseup', function (e) {
    selX = e.clientX; selY = e.clientY; onSelEnd();
  });
  document.addEventListener('touchend', function (e) {
    if (e.changedTouches[0]) {
      selX = e.changedTouches[0].clientX; selY = e.changedTouches[0].clientY;
    }
    onSelEnd();
  });
  document.addEventListener('mousedown', function (e) {
    if (!e.target.closest('#selPopup')) hideSelPopup();
  });
  document.addEventListener('scroll', hideSelPopup, { passive: true });

  // ---- search ------------------------------------------------------------
  var overlay = document.getElementById('search-overlay');
  var input = document.getElementById('search-input');
  var results = document.getElementById('search-results');
  var index = null, sel = 0, rows = [];

  function loadIndex() {
    if (index) return Promise.resolve(index);
    return fetch(ROOT + 'search-index.json')
      .then(function (r) { return r.json(); })
      .then(function (d) { index = d; return d; });
  }

  function openSearch() {
    overlay.classList.add('open');
    input.value = '';
    results.innerHTML = '<div class="empty">Type to search all documents</div>';
    input.focus();
    loadIndex();
  }
  function closeSearch() { overlay.classList.remove('open'); }

  function esc(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function highlight(text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      if (!t) return;
      out = out.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'),
                        '<mark>$1</mark>');
    });
    return out;
  }

  function snippet(text, terms) {
    var low = text.toLowerCase();
    var at = -1;
    for (var i = 0; i < terms.length; i++) {
      var p = low.indexOf(terms[i]);
      if (p !== -1 && (at === -1 || p < at)) at = p;
    }
    if (at === -1) return '';
    var start = Math.max(0, at - 70);
    var frag = text.slice(start, start + 220);
    return (start > 0 ? '…' : '') + frag + (start + 220 < text.length ? '…' : '');
  }

  function search(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length || !index) {
      results.innerHTML = '<div class="empty">Type to search all documents</div>';
      rows = [];
      return;
    }
    var hits = [];
    index.forEach(function (doc) {
      var title = doc.title.toLowerCase();
      var heads = doc.headings.join(' · ').toLowerCase();
      var text = doc.text.toLowerCase();
      var score = 0, matchedAll = true;
      terms.forEach(function (t) {
        var s = 0;
        if (title.indexOf(t) !== -1) s += 40;
        if (heads.indexOf(t) !== -1) s += 12;
        var n = text.split(t).length - 1;
        if (n) s += Math.min(n, 8);
        if (!s) matchedAll = false;
        score += s;
      });
      if (matchedAll && score > 0) hits.push({ doc: doc, score: score });
    });
    hits.sort(function (a, b) { return b.score - a.score; });
    hits = hits.slice(0, 25);
    rows = hits;
    sel = 0;

    if (!hits.length) {
      results.innerHTML = '<div class="empty">No matches for “' + esc(q) + '”</div>';
      return;
    }
    results.innerHTML = hits.map(function (h, i) {
      var sn = snippet(h.doc.text, terms);
      return '<a href="' + ROOT + h.doc.path + '" class="' + (i === 0 ? 'sel' : '') + '">'
        + '<div class="r-title">' + highlight(h.doc.title, terms) + '</div>'
        + '<div class="r-path">' + esc(h.doc.rel) + '</div>'
        + (sn ? '<div class="r-snippet">' + highlight(sn, terms) + '</div>' : '')
        + '</a>';
    }).join('');
  }

  function moveSel(delta) {
    var links = results.querySelectorAll('a');
    if (!links.length) return;
    links[sel] && links[sel].classList.remove('sel');
    sel = (sel + delta + links.length) % links.length;
    links[sel].classList.add('sel');
    links[sel].scrollIntoView({ block: 'nearest' });
  }

  if (input) {
    var t = null;
    input.addEventListener('input', function () {
      clearTimeout(t);
      var q = input.value;
      t = setTimeout(function () { loadIndex().then(function () { search(q); }); }, 90);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); moveSel(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); moveSel(-1); }
      else if (e.key === 'Enter') {
        var links = results.querySelectorAll('a');
        if (links[sel]) { e.preventDefault(); window.location = links[sel].getAttribute('href'); }
      }
    });
  }

  document.querySelectorAll('.search-btn').forEach(function (b) {
    b.addEventListener('click', openSearch);
  });
  if (overlay) overlay.addEventListener('click', function (e) {
    if (e.target === overlay) closeSearch();
  });

  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); }
    else if (e.key === 'Escape') { closeSearch(); hideSelPopup(); }
    else if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
      e.preventDefault(); openSearch();
    }
  });
})();
"""


def sidebar_html(docs: list[Doc], current: Doc | None) -> str:
    groups: dict[str, list[Doc]] = {}
    for d in docs:
        groups.setdefault(d.folder, []).append(d)

    out = ['<nav class="nav">']
    for folder in sorted(groups, key=folder_rank):
        name = folder if folder else "Overview"
        items = groups[folder]
        out.append(f'<div class="nav-group" data-group="{html.escape(folder)}">')
        out.append(
            f'<button class="group-name" type="button" aria-expanded="true">'
            f'<span class="chev" aria-hidden="true">▾</span>'
            f'<span class="group-label">{html.escape(name)}</span>'
            f'<span class="group-count">{len(items)}</span></button>'
        )
        out.append('<div class="group-items">')
        for d in items:
            active = " active" if current and d.rel == current.rel else ""
            num = re.match(r"^(\d+)", d.path.stem)
            prefix = f'<span class="num">{num.group(1)}</span>' if num else ""
            suffix = " ⓘ" if d.is_notebook else ""
            out.append(
                f'<a class="{active.strip()}" href="{d.url}">{prefix}'
                f"{html.escape(d.label)}{suffix}</a>"
            )
        out.append("</div></div>")
    out.append("</nav>")
    return "\n".join(out)


def page(docs: list[Doc], doc: Doc, prev: Doc | None, nxt: Doc | None) -> str:
    toc = ""
    if doc.toc:
        items = "".join(
            f'<a class="{"sub" if e.get("indent") else ""}" href="#{e["id"]}">'
            f'{html.escape(e["name"])}</a>'
            for e in doc.toc
        )
        toc = f'<aside class="toc"><h2>On this page</h2>{items}</aside>'

    pager = ""
    if prev or nxt:
        left = (
            f'<a href="{prev.url}"><span class="dir">← Previous</span>'
            f'<span class="name">{html.escape(prev.title)}</span></a>'
            if prev
            else "<span style='flex:1'></span>"
        )
        right = (
            f'<a class="next" href="{nxt.url}"><span class="dir">Next →</span>'
            f'<span class="name">{html.escape(nxt.title)}</span></a>'
            if nxt
            else "<span style='flex:1'></span>"
        )
        pager = f'<div class="pager">{left}{right}</div>'

    crumb = html.escape(doc.rel)

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(doc.title)} · Interview Prep</title>
<style>{CSS}</style>
</head>
<body data-root="/">
<button class="menu-toggle" aria-label="Toggle navigation">☰</button>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <h1>Interview Prep</h1>
    </div>
    <button class="search-btn">🔍 Search docs <kbd>Ctrl K</kbd></button>
    {sidebar_html(docs, doc)}
    <div class="sidebar-foot">
      <button class="icon-btn" id="theme-btn">Dark mode</button>
      <button class="icon-btn" id="toggle-all-btn">Collapse all</button>
      <span style="font-size:11px;color:var(--text-faint);margin-left:auto">
        {len(docs)}
      </span>
    </div>
  </aside>
  <main class="main">
    <div class="column">
      <div class="article-wrap">
        <div class="crumbs">{crumb}</div>
        <article>{doc.body}</article>
        {pager}
      </div>
      {toc}
    </div>
  </main>
</div>
<div class="overlay" id="search-overlay">
  <div class="palette">
    <input id="search-input" type="text" placeholder="Search all documents…"
           autocomplete="off" spellcheck="false">
    <div class="results" id="search-results"></div>
  </div>
</div>
<script>{JS}</script>
</body>
</html>"""


# ------------------------------------------------------------------------ server

class Library:
    def __init__(self):
        self.docs: list[Doc] = []
        self.by_rel: dict[str, Doc] = {}
        self.lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        docs = discover()
        for d in docs:
            render(d)
        with self.lock:
            self.docs = docs
            self.by_rel = {d.rel: d for d in docs}

    def neighbours(self, doc: Doc) -> tuple[Doc | None, Doc | None]:
        i = self.docs.index(doc)
        return (self.docs[i - 1] if i > 0 else None,
                self.docs[i + 1] if i < len(self.docs) - 1 else None)

    def index_json(self) -> bytes:
        for d in self.docs:  # cheap: mtime check skips unchanged docs
            render(d)
        payload = [
            {
                "rel": d.rel,
                "path": "doc/" + d.rel,
                "title": d.title,
                "headings": [e["name"] for e in d.toc],
                "text": d.plain[:60000],
            }
            for d in self.docs
        ]
        return json.dumps(payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    library: Library = None  # type: ignore[assignment]
    server_version = "PrepReader/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = unquote(urlparse(self.path).path)
        lib = self.library

        if path == "/":
            first = lib.docs[0] if lib.docs else None
            if not first:
                self._send(b"<h1>No .md or .ipynb files found</h1>", "text/html; charset=utf-8", 404)
                return
            self.send_response(302)
            self.send_header("Location", first.url)
            self.end_headers()
            return

        if path == "/search-index.json":
            self._send(lib.index_json(), "application/json; charset=utf-8")
            return

        if path == "/reload":
            lib.reload()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if path.startswith("/doc/"):
            rel = path[len("/doc/"):]

            # An asset (figure) or source file referenced relatively from a document.
            suffix = posixpath.splitext(rel)[1].lower()
            if suffix in ASSET_TYPES or suffix in TEXT_TYPES:
                target = (ROOT / rel).resolve()
                try:                      # refuse anything outside the docs root
                    target.relative_to(ROOT)
                except ValueError:
                    self._send(b"forbidden", "text/plain; charset=utf-8", 403)
                    return
                if not target.is_file():
                    self._send(b"not found", "text/plain; charset=utf-8", 404)
                    return
                ctype = ASSET_TYPES.get(suffix, "text/plain; charset=utf-8")
                self._send(target.read_bytes(), ctype)
                return

            doc = lib.by_rel.get(rel)
            if doc is None:
                self._send(
                    b"<h1>404</h1><p>Unknown document. <a href='/'>Back</a></p>",
                    "text/html; charset=utf-8",
                    404,
                )
                return
            # Re-render on each request so edits show up on refresh.
            render(doc)
            prev, nxt = lib.neighbours(doc)
            self._send(page(lib.docs, doc, prev, nxt).encode("utf-8"),
                       "text/html; charset=utf-8")
            return

        self._send(b"<h1>404</h1>", "text/html; charset=utf-8", 404)

    do_HEAD = do_GET


class HTTPServerV6(ThreadingHTTPServer):
    """IPv6 twin of the main server.

    On Windows "localhost" usually resolves to ::1 first; if nothing is listening
    there, every request pays a ~2s connect timeout before falling back to IPv4.
    Listening on both loopback addresses keeps the reader instant either way.
    """

    address_family = socket.AF_INET6
    daemon_threads = True


def pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port found near {preferred}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local reader for the interview-prep docs.")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    library = Library()
    if not library.docs:
        raise SystemExit(f"No .md or .ipynb files found under {ROOT}")

    port = pick_port(args.port)
    Handler.library = library
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    # Also answer on ::1 so "localhost" is fast however the browser resolves it.
    server_v6 = None
    try:
        server_v6 = HTTPServerV6(("::1", port), Handler)
        threading.Thread(target=server_v6.serve_forever, daemon=True).start()
    except OSError:
        pass  # no IPv6 loopback available; IPv4 alone is fine

    url = f"http://localhost:{port}/"
    print(f"  Interview-prep reader -> {url}")
    print(f"  {len(library.docs)} documents from {ROOT}")
    print("  Ctrl+K to search | /reload to re-scan for new files | Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        server.shutdown()
        if server_v6 is not None:
            server_v6.shutdown()


if __name__ == "__main__":
    main()
