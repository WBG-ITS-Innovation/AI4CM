"""What counts as user-visible copy, and where in the source it sits.

Why a helper rather than a regex in each test
---------------------------------------------
"Every visible string obeys the house style" is only checkable if "visible" has a
definition. Grepping the file cannot tell a sentence a Treasury reader will read from a
comment explaining why a threshold is 1.15, and the comments in this project are long and
deliberately full of the punctuation the copy rules forbid.

So this parses each page and collects string literals that reach a reader: the arguments
of the Streamlit calls that render text, the ``help=`` tooltips, and the arguments of this
project's own presentation helpers. Comments and docstrings are excluded by construction,
because the parser never sees a comment and a docstring is not a call argument.

It also records, for each string, whether it sits inside a ``with st.expander(...)``
block. That is what lets "long explanations belong behind an expander" be a test rather
than a preference.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

#: Streamlit calls whose arguments a reader sees.
STREAMLIT_TEXT_CALLS = {
    "markdown", "caption", "info", "warning", "error", "success", "write", "header",
    "subheader", "title", "text", "metric", "radio", "selectbox", "checkbox", "slider",
    "text_input", "text_area", "number_input", "multiselect", "button", "file_uploader",
    "page_link", "expander", "toggle", "download_button", "form_submit_button", "spinner",
    "dataframe", "plotly_chart", "columns", "tabs", "help",
}

#: Calls whose arguments a reader sees but which are not prose: a shell command in a code
#: block legitimately contains ``--upgrade``, and forbidding it there would mean rewriting
#: the command rather than the copy.
VERBATIM_CALLS = {"code", "json"}

#: This project's own presentation helpers, which render their arguments as page copy.
PROJECT_TEXT_HELPERS = {
    "page_header", "section_header", "callout_box", "reading_this_chart", "info_tip",
    "empty_state", "render_app_header", "gate_badge_tri", "page_intro", "term_help",
    "glossary_note",
}

#: Keyword arguments that carry copy rather than configuration.
TEXT_KEYWORDS = {"help", "label", "text", "body", "caption", "title", "placeholder"}


#: Strings made only of layout characters: a horizontal rule, a markdown table separator,
#: a column of dashes. They are markup a reader sees as a line, not a sentence they read.
_MARKUP_ONLY = set("-|_*= \n\t:#")


@dataclass(frozen=True)
class Copy:
    """One user-visible string, with enough context to report it usefully."""

    path: Path
    line: int
    text: str
    call: str
    in_expander: bool

    def where(self) -> str:
        return f"{self.path.name}:{self.line} in st.{self.call}()"

    @property
    def is_markup(self) -> bool:
        return not (set(self.text) - _MARKUP_ONLY)

    @property
    def prose_only(self) -> str:
        """The text with whole markup lines dropped.

        A multi-paragraph block can contain a markdown table, whose separator row is
        ``|---|---|``. That is a line a reader sees as a rule under a header, not a double
        hyphen in a sentence, and the punctuation rules are about sentences.
        """
        keep = [line for line in self.text.splitlines()
                if set(line) - _MARKUP_ONLY]
        return "\n".join(keep)


def _expander_line_ranges(tree: ast.AST) -> List[Tuple[int, int]]:
    """Line spans of every ``with st.expander(...)`` body.

    The expander's own label is deliberately outside the span: a label is a heading a
    reader sees before deciding to open it, so it is held to the same brevity as any other
    heading rather than to the rules for what is hidden behind one.
    """
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            name = ""
            if isinstance(call, ast.Call):
                func = call.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("expander", "popover"):
                body = node.body
                if body:
                    spans.append((body[0].lineno, max(_last_line(n) for n in body)))
    return spans


def _last_line(node: ast.AST) -> int:
    return max((getattr(n, "lineno", 0) for n in ast.walk(node)), default=0)


def _concatenated(node: ast.AST) -> str:
    """One string for an addition chain, with non-literal operands as ``{}``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _concatenated(node.left) + _concatenated(node.right)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
                       else "{}" for v in node.values)
    return "{}"


def _joined(node: ast.AST) -> List[Tuple[int, str]]:
    """Every string constant under ``node``, with implicit concatenation rejoined.

    Copy in this project is written as adjacent literals across several lines, so a
    sentence a reader sees as one string is several in the source. Checking them
    separately would flag a sentence for ending without a full stop because the line
    happened to wrap there.
    """
    out: List[Tuple[int, str]] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [(node.lineno, node.value)]
    if isinstance(node, ast.JoinedStr):
        parts, line = [], getattr(node, "lineno", 0)
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{}")          # a placeholder, so length stays honest
        return [(line, "".join(parts))]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # `"text " + pub["reason"] + " more text"` is one sentence a reader sees. Each
        # operand is rendered strictly: a literal contributes its text and anything else
        # contributes a placeholder, exactly as an f-string's interpolations do. Walking
        # into a non-literal operand would pull out incidental strings, and a dictionary
        # key is not copy: `pub["reason_plain"]` would otherwise append the words
        # "reason_plain" to the sentence and make it look unfinished.
        return [(getattr(node, "lineno", 0), _concatenated(node))]
    for child in ast.iter_child_nodes(node):
        out.extend(_joined(child))
    return out


def visible_copy(path: Path) -> List[Copy]:
    """Every user-visible string in one page or helper module."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    spans = _expander_line_ranges(tree)

    def inside(line: int) -> bool:
        return any(lo <= line <= hi for lo, hi in spans)

    out: List[Copy] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in VERBATIM_CALLS:
            continue
        if name not in STREAMLIT_TEXT_CALLS and name not in PROJECT_TEXT_HELPERS:
            continue
        carriers = list(node.args)
        carriers += [kw.value for kw in node.keywords if kw.arg in TEXT_KEYWORDS]
        for carrier in carriers:
            # `st.markdown(section_header("Title", "subtitle"))` is one visible thing, not
            # two. Attributing its strings to the outer markdown as well as to the inner
            # helper would hold a section subtitle to the rules for a paragraph, and a
            # subtitle is a heading. The inner call is collected on its own pass.
            if isinstance(carrier, ast.Call):
                inner = carrier.func
                inner_name = (inner.attr if isinstance(inner, ast.Attribute)
                              else getattr(inner, "id", ""))
                if inner_name in PROJECT_TEXT_HELPERS or inner_name in STREAMLIT_TEXT_CALLS:
                    continue
            for line, text in _joined(carrier):
                if not text.strip():
                    continue
                candidate = Copy(path=path, line=line, text=text, call=name,
                                 in_expander=inside(line))
                if candidate.is_markup:
                    continue
                out.append(candidate)
    return sorted(out, key=lambda c: c.line)


def pages(frontend: Path) -> List[Path]:
    """Every page a reader can open, plus the entry point."""
    return sorted(frontend.glob("pages/*.py")) + [frontend / "Overview.py"]


def helper_modules(frontend: Path) -> List[Path]:
    """Modules that hold copy the pages render."""
    return [frontend / name for name in
            ("ui_styles.py", "format_gel.py", "recommender.py", "intervals.py",
             "data_preflight.py", "exploratory.py", "run_errors.py")
            if (frontend / name).exists()]


def all_text_of(path: Path) -> str:
    """One blob of a file's visible copy, for presence checks."""
    return "\n".join(c.text for c in visible_copy(path))


def terms_used(text: str, terms: Set[str]) -> Set[str]:
    """Which glossary terms appear in ``text``, matched case-insensitively."""
    lowered = text.lower()
    return {t for t in terms if t.lower() in lowered}
