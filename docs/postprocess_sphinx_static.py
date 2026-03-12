#!/usr/bin/env python3
"""
Post-process Sphinx HTML output to avoid underscore-prefixed static paths.

Many static hosting pipelines (and Jekyll in particular) may omit directories
starting with '_' (e.g. Sphinx's default `_static/`). This script:
1) Copies `<outdir>/_static/` -> `<outdir>/static/` (rsync-like).
2) Rewrites HTML references from `_static/` to `static/`.

This makes the built docs work even when `_static/` is not served.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _copytree_overwrite(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _rewrite_html_static_refs(outdir: Path) -> int:
    replaced_files = 0
    for p in outdir.rglob("*.html"):
        # Don't bother rewriting the duplicated static payload
        if "/_static/" in p.as_posix() or "/static/" in p.as_posix():
            continue
        s = p.read_text(encoding="utf-8", errors="ignore")
        if "_static/" not in s:
            continue
        s2 = s.replace("_static/", "static/")
        if s2 != s:
            p.write_text(s2, encoding="utf-8")
            replaced_files += 1
    return replaced_files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        required=True,
        help="Sphinx HTML output directory (contains _static/ and HTML files).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    src = outdir / "_static"
    dst = outdir / "static"

    _copytree_overwrite(src, dst)
    _rewrite_html_static_refs(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

