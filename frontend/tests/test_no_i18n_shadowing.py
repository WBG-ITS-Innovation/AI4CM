"""No page may rebind the name it imported its translator under.

The bug this exists to prevent
------------------------------
`pages/01_Forecast.py` imported the translator as `from i18n import t as _t`, then used `_t` as
a loop and assignment variable for a Treasury line name in three places. Two of those sit
inside a module-level `if`, so they rebound the module global rather than a local: the
translator became a string, and the next call to it raised

    TypeError: 'str' object is not callable

on render, taking the whole page down. It reached a running app because it is invisible to
review (the import and the reuse are 550 lines apart) and invisible to the page's own render
test (AppTest runs a page at default widget values, and the shadowing line only executed after
a reader clicked one radio).

Why a source check rather than a render check
--------------------------------------------
A render test catches this only on the widget path it happens to drive, which is exactly what
let it through. Shadowing is a property of the source, so the source is what to check: it holds
for every page, on every path, including paths no test drives.

Why the check is scope-aware
----------------------------
The first version of this lint compared the imported names against every assignment anywhere in
the file, and it reported `ui_styles.py`, wrongly. That module imports the translator *inside a
function*, so the name is a local there, and its two `t` bindings are comprehension variables,
which in Python 3 have a scope of their own and cannot reach anything outside it. Neither can
rebind the other. A lint that cries wolf on correct code gets deleted, so this one compares
bindings only within the scope that holds the import, plus the one case where a nested scope
genuinely can reach out: an explicit `global`.

This is a narrow lint on purpose. It does not ban reusing a name generally. It bans reusing the
specific names a scope imported from `i18n`, which are the names whose rebinding turns a working
page into a stack trace.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

FRONTEND = Path(__file__).resolve().parents[1]

#: Every page, plus the entry point and the helper modules. `ui_styles` also imports the
#: translator, and the same mistake there would break every page at once rather than one.
SOURCES = sorted(FRONTEND.glob("pages/*.py")) + sorted(FRONTEND.glob("*.py"))

#: Nodes that open a new scope. A name bound inside one of these does not rebind a same-named
#: import in the scope outside it, which is the distinction the first version of this lint
#: missed. Comprehensions are included: Python 3 gives each its own scope.
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _child_scopes(scope: ast.AST) -> List[ast.AST]:
    """Scope-opening nodes directly inside ``scope``, not nested deeper."""
    found: List[ast.AST] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                found.append(child)
            else:
                walk(child)

    walk(scope)
    return found


def _own_nodes(scope: ast.AST):
    """Every node belonging to ``scope`` itself, stopping at any nested scope.

    A module-level `if`, `for`, `try` or `with` is NOT a scope, so its bodies are included.
    That matters: the real defect lived inside a module-level `if`, where an assignment
    rebinds the module global exactly as a top-level one would.
    """
    def walk(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                continue
            yield child
            yield from walk(child)

    yield from walk(scope)


def _imported_from_i18n(scope: ast.AST) -> Dict[str, str]:
    """``{local name: original name}`` for what this scope pulls out of ``i18n`` itself."""
    names: Dict[str, str] = {}
    for node in _own_nodes(scope):
        if isinstance(node, ast.ImportFrom) and node.module == "i18n":
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
    return names


def _names_bound(scope: ast.AST) -> Set[str]:
    """Names this scope binds itself: assignment, `for`, `with ... as`, `except ... as`.

    Excludes `ast.ImportFrom`, which is the binding being checked against, and excludes
    function parameters, which are genuine locals of a nested scope.
    """
    out: Set[str] = set()

    def record(target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                out.add(node.id)

    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                record(t)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            record(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            record(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            record(node.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
    return out


def _globals_rebound_in_nested_scopes(module: ast.AST) -> Set[str]:
    """Names a function declares `global` and then assigns.

    The one way a nested scope genuinely reaches a module-level import. Rare, and the whole
    point of the lint is that this failure mode is not obvious, so it is checked rather than
    assumed absent.
    """
    out: Set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = {n for stmt in ast.walk(node) if isinstance(stmt, ast.Global)
                    for n in stmt.names}
        if declared:
            out |= declared & _names_bound(node)
    return out


def _clashes(source: str) -> List[Tuple[str, str, str]]:
    """``(scope description, local name, i18n name)`` for every real shadowing."""
    module = ast.parse(source)
    found: List[Tuple[str, str, str]] = []

    def describe(scope: ast.AST) -> str:
        return "module level" if isinstance(scope, ast.Module) else f"{getattr(scope, 'name', '<lambda>')}()"

    def visit(scope: ast.AST) -> None:
        imported = _imported_from_i18n(scope)
        if imported:
            bound = _names_bound(scope)
            if isinstance(scope, ast.Module):
                bound |= _globals_rebound_in_nested_scopes(scope)
            for local in sorted(set(imported) & bound):
                found.append((describe(scope), local, imported[local]))
        for child in _child_scopes(scope):
            visit(child)

    visit(module)
    return found


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_page_rebinds_a_name_it_imported_from_i18n(path: Path):
    source = path.read_text(encoding="utf-8")
    if "i18n" not in source:
        pytest.skip(f"{path.name} does not import from i18n")

    clashes = _clashes(source)
    assert not clashes, "\n".join(
        [f"{path.name} shadows its own i18n import:"]
        + [f"  at {where}: imported i18n.{orig} as {local}, then assigned to {local}"
           for where, local, orig in clashes]
        + ["At module level that rebinds the global, so a later call raises",
           "TypeError: 'str' object is not callable. Rename the variable, or import the",
           "translator under a name nothing would reuse (`_translate` is the convention here)."]
    )


# ══════════════════════════════════════════════════════════════════════════════
# The lint's own tests. A lint nobody has watched fail is a lint nobody knows works, and this
# one has already been wrong once: its first version reported `ui_styles.py`, which is correct
# code. So both directions are pinned, and the false positive it produced is a case of its own.
# ══════════════════════════════════════════════════════════════════════════════

def test_it_catches_the_real_defect():
    """The shape of the actual bug: a module-level `if` rebinding a module-level import."""
    assert _clashes(
        "from i18n import t as _t\n"
        "if True:\n"
        "    for _t in ['Revenues']:\n"
        "        print(_t)\n"
    ) == [("module level", "_t", "t")]


def test_it_catches_a_plain_module_level_assignment():
    assert _clashes("from i18n import t as _t\n_t = 'Revenues'\n") == [("module level", "_t", "t")]


def test_it_catches_a_global_rebound_from_inside_a_function():
    """The one way a nested scope can reach a module-level name."""
    assert _clashes(
        "from i18n import t as _t\n"
        "def f():\n"
        "    global _t\n"
        "    _t = 'Revenues'\n"
    ) == [("module level", "_t", "t")]


def test_it_accepts_the_fix_that_was_applied():
    assert not _clashes(
        "from i18n import t as _translate\n"
        "if True:\n"
        "    for _tgt in ['Revenues']:\n"
        "        print(_translate(_tgt))\n"
    )


def test_it_does_not_report_a_comprehension_variable():
    """The false positive its first version produced, as `ui_styles.py` actually reads.

    A comprehension has its own scope in Python 3, so `t` here cannot touch the `t` imported
    in the function above it. Both are correct code and neither may be reported.
    """
    assert not _clashes(
        "GLOSSARY = {}\n"
        "def _translate(text):\n"
        "    from i18n import t\n"
        "    return t(text)\n"
        "def note(terms):\n"
        "    return '  '.join(_translate(GLOSSARY[t]) for t in terms if t in GLOSSARY)\n"
    )


def test_it_does_not_report_a_function_local_that_shares_a_name_with_a_module_import():
    """A local binding in a nested scope leaves the module global alone."""
    assert not _clashes(
        "from i18n import t as _t\n"
        "def f():\n"
        "    _t = 'Revenues'\n"
        "    return _t\n"
    )
