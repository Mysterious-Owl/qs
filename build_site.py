#!/usr/bin/env python3
"""Build a static site from the docs — same look as `reader.py`, no server needed.

    python build_site.py                    # -> ./_site, ready for GitHub Pages
    python build_site.py --out docs         # different output folder
    python build_site.py --exclude "Example_Company/*"   # leave a folder out
    python build_site.py --clean            # wipe the output folder first

Reuses reader.py's markdown rendering, CSS and JS, so the published pages look and
behave like the local reader: sidebar, per-page table of contents, Ctrl+K search,
light/dark themes.

Every URL it writes is *relative*, so the site works both at a domain root and
under a project subpath like https://<user>.github.io/<repo>/.

PRIVACY: this publishes whatever it builds. Only files actually linked from a
document are copied, so stray files (a CV PDF, scratch notes) are not swept in —
but the documents themselves go up verbatim. Review what is in them, and remember
GitHub Pages on a public repo is public to everyone. Use --exclude to leave
individual documents out.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    import reader
except ImportError:  # pragma: no cover
    sys.exit("build_site.py must sit next to reader.py")

ROOT = reader.ROOT

# The document that becomes the site's landing page. It is published as
# index.html *only* — emitting README.html too would be a byte-identical
# duplicate, and every link would then have two valid targets.
LANDING_REL: str | None = None


# ----------------------------------------------------------------- link helpers

def out_rel(rel: str) -> str:
    """Source path (a.md / a.ipynb) -> published path (a.html)."""
    if LANDING_REL is not None and rel == LANDING_REL:
        return "index.html"
    stem, ext = posixpath.splitext(rel)
    return stem + ".html" if ext in (".md", ".ipynb") else rel


def root_prefix(rel: str) -> str:
    """Relative prefix from a page back to the site root ('', '../', '../../')."""
    depth = rel.count("/")
    return "../" * depth


def rewrite_links_static(body: str, doc: reader.Doc, wanted: set[str],
                         published: set[str], dead: list[str]) -> str:
    """Rewrite in-body links for the static site.

    Document links keep their relative form but change extension to .html.
    Asset links are left alone and recorded in `wanted` so they get copied.
    A link to a document that exists on disk but was excluded from this build is
    marked dead rather than left pointing at a page that will not be published.
    """

    def resolve(href: str):
        target, _, fragment = href.partition("#")
        if not target:
            return None
        resolved = posixpath.normpath(posixpath.join(doc.folder, unquote(target)))
        return resolved, fragment

    def fix_href(match: re.Match) -> str:
        quote, href = match.group(1), match.group(2)
        parsed = urlparse(href)
        if parsed.scheme or href.startswith(("#", "/", "mailto:")):
            if parsed.scheme in ("http", "https"):
                return f'target="_blank" rel="noopener" href={quote}{href}{quote}'
            return match.group(0)

        got = resolve(href)
        if got is None:
            return match.group(0)
        resolved, fragment = got
        candidate = ROOT / resolved

        if candidate.is_dir():
            readme = candidate / "README.md"
            if not readme.exists():
                dead.append(f"{doc.rel} -> {href} (folder has no README.md)")
                return f'href={quote}#{quote} class="dead-link"'
            resolved = readme.relative_to(ROOT).as_posix()
        elif not candidate.exists():
            dead.append(f"{doc.rel} -> {href} (missing)")
            return f'href={quote}#{quote} class="dead-link"'

        suffix = "#" + fragment if fragment else ""
        ext = posixpath.splitext(resolved)[1].lower()
        if ext in (".md", ".ipynb"):
            if resolved not in published:      # excluded from this build
                return f'href={quote}#{quote} class="dead-link"'   # expected: --exclude
            new = posixpath.relpath(out_rel(resolved), doc.folder or ".")
        else:
            wanted.add(resolved)          # asset or source file: copy it verbatim
            new = posixpath.relpath(resolved, doc.folder or ".")
        return f"href={quote}{new}{suffix}{quote}"

    def fix_src(match: re.Match) -> str:
        quote, src = match.group(1), match.group(2)
        if urlparse(src).scheme or src.startswith(("/", "data:")):
            return match.group(0)
        got = resolve(src)
        if got is None:
            return match.group(0)
        resolved, _ = got
        if (ROOT / resolved).exists():
            wanted.add(resolved)
        return match.group(0)             # already correctly relative

    body = re.sub(r'href=(["\'])([^"\']+)\1', fix_href, body)
    body = re.sub(r'src=(["\'])([^"\']+)\1', fix_src, body)
    return body


# --------------------------------------------------------------------- page HTML

def sidebar_html(docs: list[reader.Doc], current: reader.Doc) -> str:
    groups: dict[str, list[reader.Doc]] = {}
    for d in docs:
        groups.setdefault(d.folder, []).append(d)

    here = current.folder or "."
    out = ['<nav class="nav">']
    # reader.folder_rank keeps the server and the published site in one order.
    for folder in sorted(groups, key=reader.folder_rank):
        items = groups[folder]
        out.append(f'<div class="nav-group" data-group="{html.escape(folder)}">')
        out.append(
            f'<button class="group-name" type="button" aria-expanded="true">'
            f'<span class="chev" aria-hidden="true">&#9662;</span>'
            f'<span class="group-label">{html.escape(folder or "Overview")}</span>'
            f'<span class="group-count">{len(items)}</span></button>'
        )
        out.append('<div class="group-items">')
        for d in items:
            href = posixpath.relpath(out_rel(d.rel), here)
            active = " active" if d.rel == current.rel else ""
            num = re.match(r"^(\d+)", d.path.stem)
            prefix = f'<span class="num">{num.group(1)}</span>' if num else ""
            suffix = " &#9432;" if d.is_notebook else ""
            label = "Home" if d.rel == LANDING_REL else d.label
            out.append(
                f'<a class="{active.strip()}" href="{href}">{prefix}'
                f"{html.escape(label)}{suffix}</a>"
            )
        out.append("</div></div>")
    out.append("</nav>")
    return "\n".join(out)


def page(docs, doc, prev, nxt, site_title: str) -> str:
    base = root_prefix(doc.rel) or ""

    toc = ""
    if doc.toc:
        items = "".join(
            f'<a class="{"sub" if e.get("indent") else ""}" href="#{e["id"]}">'
            f'{html.escape(e["name"])}</a>'
            for e in doc.toc
        )
        toc = f'<aside class="toc"><h2>On this page</h2>{items}</aside>'

    here = doc.folder or "."
    pager = ""
    if prev or nxt:
        left = (
            f'<a href="{posixpath.relpath(out_rel(prev.rel), here)}">'
            f'<span class="dir">&larr; Previous</span>'
            f'<span class="name">{html.escape(prev.title)}</span></a>'
            if prev else "<span style='flex:1'></span>"
        )
        right = (
            f'<a class="next" href="{posixpath.relpath(out_rel(nxt.rel), here)}">'
            f'<span class="dir">Next &rarr;</span>'
            f'<span class="name">{html.escape(nxt.title)}</span></a>'
            if nxt else "<span style='flex:1'></span>"
        )
        pager = f'<div class="pager">{left}{right}</div>'

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(doc.title)} &middot; {html.escape(site_title)}</title>
<meta name="description" content="{html.escape(doc.plain[:150])}">
<style>{reader.CSS}</style>
</head>
<body data-root="{base}">
<button class="menu-toggle" aria-label="Toggle navigation">&#9776;</button>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <h1><a href="{base}index.html" style="color:inherit">{html.escape(site_title)}</a></h1>
    </div>
    <button class="search-btn">&#128269; Search docs <kbd>Ctrl K</kbd></button>
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
        <div class="crumbs">{html.escape(doc.rel)}</div>
        <article>{doc.body}</article>
        {pager}
      </div>
      {toc}
    </div>
  </main>
</div>
<div class="overlay" id="search-overlay">
  <div class="palette">
    <input id="search-input" type="text" placeholder="Search all documents&hellip;"
           autocomplete="off" spellcheck="false">
    <div class="results" id="search-results"></div>
  </div>
</div>
<script>{reader.JS}</script>
</body>
</html>"""


NOT_FOUND = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not found</title>
<style>{css}</style>
</head>
<body data-root="">
<div style="max-width:36rem;margin:18vh auto;padding:0 24px;font-family:var(--sans)">
  <h1 style="font-family:var(--serif);font-size:2rem;margin:0 0 10px">404</h1>
  <p style="color:var(--text-dim)">That page isn't here.</p>
  <p><a id="home" href="./">Back to the docs</a></p>
</div>
<script>
// A project site lives under /<repo>/, so derive the home link from the path.
(function () {{
  var parts = location.pathname.split('/').filter(Boolean);
  var base = (location.hostname.endsWith('github.io') && parts.length) ? '/' + parts[0] + '/' : '/';
  document.getElementById('home').setAttribute('href', base);
}})();
</script>
</body>
</html>"""


# ------------------------------------------------------------------------ build

def build(out_dir: Path, excludes: list[str], site_title: str, clean: bool) -> int:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Never let a previous build become input to the next one.
    reader.SKIP_DIRS.add(out_dir.name)

    docs = [
        d for d in reader.discover()
        if not any(fnmatch.fnmatch(d.rel, pat) for pat in excludes)
    ]
    if not docs:
        sys.exit("No documents to build (everything excluded?)")

    # Decide the landing page up front: out_rel() and every link depend on it.
    global LANDING_REL
    landing = next((d for d in docs if d.rel.lower() == "readme.md"), docs[0])
    LANDING_REL = landing.rel

    # reader.render() rewrites links to the dev server's absolute /doc/ routes.
    # Those break under a GitHub Pages project subpath, so neutralise that step
    # and apply the relative rewrite below instead.
    original_rewrite = reader.rewrite_links
    reader.rewrite_links = lambda body, doc: body
    try:
        for d in docs:
            reader.render(d, force=True)
    finally:
        reader.rewrite_links = original_rewrite

    wanted: set[str] = set()
    dead: list[str] = []
    published = {d.rel for d in docs}
    index = []

    for i, doc in enumerate(docs):
        doc.body = rewrite_links_static(doc.body, doc, wanted, published, dead)
        prev = docs[i - 1] if i > 0 else None
        nxt = docs[i + 1] if i < len(docs) - 1 else None

        dest = out_dir / out_rel(doc.rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page(docs, doc, prev, nxt, site_title), encoding="utf-8")

        index.append({
            "rel": doc.rel,
            "path": out_rel(doc.rel),
            "title": doc.title,
            "headings": [e["name"] for e in doc.toc],
            "text": doc.plain[:60000],
        })

    (out_dir / "search-index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")

    # Only assets a document actually links to — nothing else is published.
    copied = 0
    for rel in sorted(wanted):
        src = ROOT / rel
        if not src.is_file():
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied += 1

    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    (out_dir / "404.html").write_text(NOT_FOUND.format(css=reader.CSS), encoding="utf-8")

    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  {len(docs)} pages + {copied} assets -> {out_dir}")
    print(f"  {size / 1024 / 1024:.1f} MB total")
    if excludes:
        print(f"  excluded: {', '.join(excludes)}")
    if dead:
        print(f"  WARNING: {len(dead)} dead link(s) not explained by --exclude:")
        for d in dead[:10]:
            print(f"    {d}")
    return len(docs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a static site from the docs.")
    ap.add_argument("--out", default="_site",
                    help="output folder (default: _site)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob of source paths to leave out; repeatable")
    ap.add_argument("--title", default="Interview Prep", help="site title")
    ap.add_argument("--clean", action="store_true", help="wipe the output folder first")
    args = ap.parse_args()

    out_dir = (ROOT / args.out).resolve()
    if out_dir == ROOT:
        sys.exit("--out must not be the repository root")

    print(f"Building static site from {ROOT}")
    build(out_dir, args.exclude, args.title, args.clean)
    print("\n  Preview locally:")
    print(f"    python -m http.server -d {args.out} 8080")
    print("\n  Publish: just push to main -- .github/workflows/pages.yml builds")
    print("    and deploys this for you (Pages source: GitHub Actions).")


if __name__ == "__main__":
    main()
