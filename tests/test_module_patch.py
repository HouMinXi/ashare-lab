"""Guard the sys.modules patching helper.

These use a synthetic package so the outcome does not depend on which
ashare modules an earlier test happened to import.

The defect needs one asymmetry: the parent package is already loaded when
the window opens, so a snapshot restore brings it back carrying its
attribute, while the child imported inside the window is dropped. Import
both inside the window and nothing is left behind.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tests.module_patch import patched_modules

PKG = "probepkg"
LEAF = "probepkg.leaf"


@pytest.fixture
def synthetic_pkg(tmp_path, monkeypatch):
    """Put a two-level package on sys.path, absent from sys.modules."""
    root = tmp_path / "pkgroot"
    pkg = root / PKG
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "leaf.py").write_text("value = 'real'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    assert PKG not in sys.modules
    # load the parent only, which is the shape a real package is in
    importlib.import_module(PKG)
    assert LEAF not in sys.modules
    yield
    for name in [n for n in list(sys.modules) if n.startswith(PKG)]:
        sys.modules.pop(name, None)


def test_restores_an_overridden_entry():
    """An entry that existed before must come back as the same object."""
    before = sys.modules["json"]
    with patched_modules({"json": object()}):
        assert sys.modules["json"] is not before
    assert sys.modules["json"] is before


def test_removes_an_entry_that_was_absent():
    """An override for a name that was not loaded must not linger."""
    name = "probe_absent_module_name"
    assert name not in sys.modules
    with patched_modules({name: object()}):
        assert name in sys.modules
    assert name not in sys.modules


def test_forgets_a_module_imported_inside_the_window(synthetic_pkg):
    """A module that imported while a fake was installed may have captured
    it, so it must not stay cached once the window closes."""
    with patched_modules({"json": object()}):
        importlib.import_module(LEAF)
        assert LEAF in sys.modules
    assert LEAF not in sys.modules


def test_leaves_no_orphan_attribute_on_the_parent(synthetic_pkg):
    """Forgetting a child in sys.modules while its parent keeps pointing at
    it is the state that silently defeats a later monkeypatch."""
    with patched_modules({"json": object()}):
        importlib.import_module(LEAF)
    parent = sys.modules[PKG]
    assert not hasattr(parent, "leaf"), (
        "the parent still reaches a module sys.modules has dropped"
    )


def test_a_later_monkeypatch_reaches_a_lazily_imported_name(
    synthetic_pkg, monkeypatch
):
    """pytest resolves a dotted target by walking getattr from the top
    package, so an orphan attribute would take the patch while the import
    machinery hands the code under test a fresh copy."""
    with patched_modules({"json": object()}):
        importlib.import_module(LEAF)

    monkeypatch.setattr(f"{LEAF}.value", "stubbed")

    mod = importlib.import_module(LEAF)
    assert mod.value == "stubbed", "the stub did not reach the imported module"
