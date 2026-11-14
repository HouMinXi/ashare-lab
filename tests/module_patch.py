"""Swap sys.modules entries for the duration of a block.

``patch.dict(sys.modules, ...)`` looks like the obvious tool and is not:
it restores by clearing the table and refilling it from a snapshot taken on
entry. Every module imported inside the window is therefore dropped, while
a parent package that was already loaded comes back from the snapshot still
holding an attribute that points at the dropped child.

That leaves two different answers to "what is ``a.b.c``". pytest resolves a
dotted monkeypatch target by importing the top package and then walking
``getattr``, so it finds the orphan and patches that, while the code under
test imports ``a.b.c`` and gets a freshly built module. The stub silently
misses and the real function runs, which is how a unit test ends up reading
production data.

This helper keeps the two answers in step: modules imported inside the
window are forgotten, as they may have captured whatever fake was
installed, and the parent's attribute goes with them.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator, Mapping

_ABSENT = object()


def _forget(name: str) -> None:
    """Drop ``name`` from sys.modules and from its parent package.

    Args:
        name: a dotted module name.
    """
    module = sys.modules.pop(name, None)
    if module is None or "." not in name:
        return
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, child, None) is module:
        delattr(parent, child)


@contextlib.contextmanager
def patched_modules(overrides: Mapping[str, object]) -> Iterator[None]:
    """Install ``overrides`` into sys.modules for the duration of the block.

    Args:
        overrides: module names mapped to what ``import`` should find. A
            value of None makes the import fail, which is how an absent
            dependency is simulated.

    Yields:
        None, with the overrides in place.
    """
    saved = {name: sys.modules.get(name, _ABSENT) for name in overrides}
    before = set(sys.modules)
    sys.modules.update(overrides)
    try:
        yield
    finally:
        fresh = set(sys.modules) - before - set(overrides)
        # deepest first, so a parent is still around to be corrected
        for name in sorted(fresh, key=len, reverse=True):
            _forget(name)
        for name, value in saved.items():
            if value is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
