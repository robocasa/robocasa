# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import json
import os
import sys
import importlib.util

sys.path.insert(0, os.path.abspath("."))

# --- Build-time generation ---------------------------------------------------
#
# Composite Tasks page should not "pop in" after JS runs. We generate the
# activity <details> markup at build time and include it from the page source.
#
_CONF_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPOSITE_TASK_ATTRS_JSON = os.path.join(
    _CONF_DIR, "composite_tasks", "task_attributes.json"
)
_COMPOSITE_TASKS_GEN_DIR = os.path.join(_CONF_DIR, "tasks", "_generated")
_COMPOSITE_TASKS_DETAILS_RST = os.path.join(
    _COMPOSITE_TASKS_GEN_DIR, "composite_tasks_details.rst"
)


def _rc_anchor_id_from_activity_title(title: str) -> str:
    # Match the JS slugging logic in composite_tasks_dropdown.js (roughly Sphinx defaults)
    s = (title or "").lower().replace("&", " and ")
    out = []
    for ch in s:
        if ch.isalnum() or ch in [" ", "-"]:
            out.append(ch)
    s = "".join(out).strip()
    s = "-".join([p for p in s.split() if p])
    while "--" in s:
        s = s.replace("--", "-")
    return s


def _rc_escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _rc_format_desc_html(desc: str) -> str:
    # Mirrors the JS formatting enough for nice initial render:
    # {x} -> [<strong><em>x</em></strong>]
    esc = _rc_escape_html(desc or "")
    # lightweight brace formatting
    out = []
    i = 0
    while i < len(esc):
        if esc[i] == "{":
            j = esc.find("}", i + 1)
            if j != -1:
                inner = esc[i + 1 : j]
                out.append("[<strong><em>")
                out.append(inner)
                out.append("</em></strong>]")
                i = j + 1
                continue
        out.append(esc[i])
        i += 1
    return "".join(out)


def _generate_composite_tasks_details() -> None:
    os.makedirs(_COMPOSITE_TASKS_GEN_DIR, exist_ok=True)
    if not os.path.isfile(_COMPOSITE_TASK_ATTRS_JSON):
        with open(_COMPOSITE_TASKS_DETAILS_RST, "w", encoding="utf-8") as f:
            f.write(
                ".. raw:: html\n\n   <!-- composite_tasks/task_attributes.json not found -->\n"
            )
        return

    with open(_COMPOSITE_TASK_ATTRS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    # Keep only composite (exclude the "Atomic" activity rows in this JSON)
    tasks = [
        t
        for t in tasks
        if isinstance(t, dict) and (t.get("activity") or "").strip().lower() != "atomic"
    ]

    grouped: dict[str, list[dict]] = {}
    for t in tasks:
        activity = str(t.get("activity") or "").strip()
        if not activity:
            continue
        grouped.setdefault(activity, []).append(t)

    activities_sorted = sorted(grouped.keys(), key=lambda s: s.lower())

    # Emit raw HTML via an RST raw directive (works regardless of markdown parser).
    lines: list[str] = []
    lines.append(".. raw:: html")
    lines.append("")
    lines.append(
        "   <!-- Auto-generated. Do not edit; edit composite_tasks/task_attributes.json instead. -->"
    )
    for activity in activities_sorted:
        rows = grouped.get(activity) or []
        rows_sorted = sorted(rows, key=lambda r: str(r.get("name") or ""))
        anchor_id = _rc_anchor_id_from_activity_title(activity) or _rc_escape_html(
            activity
        )
        n = len(rows_sorted)
        lines.append(f'   <details class="rc-activity" id="{anchor_id}">')
        lines.append("     <summary>")
        lines.append('       <div class="rc-activity-summary-left">')
        lines.append(
            f'         <span class="rc-activity-title">{_rc_escape_html(activity)}</span>'
        )
        # meta/category pill is added/enhanced by JS; omit here to keep generator simple
        lines.append("       </div>")
        lines.append(
            f'       <span class="rc-activity-badge">{n} task{"s" if n != 1 else ""}</span>'
        )
        lines.append("     </summary>")
        lines.append('     <div class="rc-activity-body">')
        lines.append('       <table class="docutils" border="1">')
        lines.append(
            "         <thead><tr><th>Task</th><th>Description</th></tr></thead>"
        )
        lines.append("         <tbody>")
        for t in rows_sorted:
            name = str(t.get("name") or "").strip()
            desc = str(t.get("description") or "").strip()
            if not name:
                continue
            lines.append("           <tr>")
            lines.append(f"             <td><code>{_rc_escape_html(name)}</code></td>")
            lines.append(f"             <td>{_rc_format_desc_html(desc)}</td>")
            lines.append("           </tr>")
        lines.append("         </tbody>")
        lines.append("       </table>")
        lines.append("     </div>")
        lines.append("   </details>")

    with open(_COMPOSITE_TASKS_DETAILS_RST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


_generate_composite_tasks_details()

# Optional: avoid importing robocasa so docs build without full runtime (numpy, cv2, robosuite).
robocasa = None
try:
    import robocasa  # type: ignore
except BaseException:  # noqa: BLE001
    pass


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
]

# Optional extensions (docs should still build without them)
if _has_module("sphinx_markdown_tables"):
    extensions.append("sphinx_markdown_tables")

# Markdown parser
if _has_module("recommonmark"):
    extensions.append("recommonmark")  # legacy markdown support
elif _has_module("myst_parser"):
    extensions.append("myst_parser")

if _has_module("nbsphinx"):
    extensions.append("nbsphinx")


# Sphinx-apidoc variables
apidoc_module_dir = "../robocasa"
apidoc_output_dir = "reference"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# source_parsers = {
#     '.md': CommonMarkParser,
# }

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst"]
if _has_module("recommonmark") or _has_module("myst_parser"):
    source_suffix.append(".md")
if _has_module("nbsphinx"):
    source_suffix.append(".ipynb")

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "RoboCasa"
copyright = "the RoboCasa core team, 2026"
author = "the RoboCasa core team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
if robocasa is not None and getattr(robocasa, "__version__", None):
    version = (".").join(robocasa.__version__.split(".")[:-1])
    # The full version, including alpha/beta/rc tags.
    release = (".").join(robocasa.__version__.split(".")[:-1])
else:
    version = "1.0"
    release = "1.0"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects htmlstatic_path and html_extra_path.
exclude_patterns = ["_build", "build", "_versions", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if _has_module("sphinx_book_theme"):
    html_theme = "sphinx_book_theme"
elif _has_module("pydata_sphinx_theme"):
    html_theme = "pydata_sphinx_theme"
else:
    html_theme = "alabaster"
html_logo = "robocasa_logo.svg"
html_favicon = "images/Robocasa_web_logo.svg"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Custom JS
html_js_files = [
    # NOTE: `html_static_path` pulls from multiple source directories (including
    # `atomic_tasks/` and `composite_tasks/`) and Sphinx merges them into the
    # built `_static/` root (it does *not* preserve the directory name).
    # Therefore these must be referenced as flat filenames here.
    "composite_task_attributes.js",
    "composite_episode_lengths.js",
    "composite_tasks_dropdown.js",
    "atomic_task_attributes.js",
    "atomic_episode_lengths.js",
    "atomic_task_index.js",
    "atomic_tasks.js",
]

# Custom CSS
html_css_files = [
    "theme_overrides.css",
    "composite_tasks_dropdown.css",
    "atomic_tasks.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
_confdir = os.path.dirname(os.path.abspath(__file__))
html_static_path = [
    p
    for p in ["_static", "images", "atomic_tasks", "composite_tasks"]
    if os.path.isdir(os.path.join(_confdir, p))
]
# Ensure _static exists so Sphinx doesn't warn (optional overrides go here).
os.makedirs(os.path.join(_confdir, "_static"), exist_ok=True)
if "_static" not in html_static_path:
    html_static_path.insert(0, "_static")

# html_context = {
#     'css_files': [
#         'static/theme_overrides.css',  # override wide tables in RTD theme
#     ],
# }

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "robocasadoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "robocasa.tex", "RoboCasa Documentation", author, "manual"),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "robocasa", "RoboCasa Documentation", [author], 1)]
