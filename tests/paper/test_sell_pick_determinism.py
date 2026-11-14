"""The sell pick must not depend on how a set happens to iterate.

topk_dropout_orders sorts sell candidates by signals.get(s, -inf) and
takes n_drop of them. Every held symbol missing from the signal dict gets
-inf, so two such holdings tie, and Python's sort is stable: tied entries
keep the order they arrived in. They arrive from `current_positions`,
which the pipeline passes as a set, so the choice follows per-process
string hashing.

Measured on 2026-10-22 by instrumenting the running pipeline: 134 signals,
11 held, SZ301029 and SZ301293 both absent from the signals and both
scored -inf. Under PYTHONHASHSEED=1 the set yielded SZ301029 first and it
was sold; under seed 2 SZ301293 came first and it was sold instead. The
resulting books differed by 110 yuan over three days.

These tests pin the pick to the inputs rather than to the iteration order,
and state the tie-break instead of inheriting one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from ashare_lab.config import PROJECT_ROOT
from ashare_lab.paper.signal import topk_dropout_orders

# two held symbols absent from the signals, as on 2026-10-22
TIED = ["SZ301029", "SZ301293"]
SCORED = {f"S{i:03d}": 1.0 - i * 0.01 for i in range(20)}


def _held(order: list[str]) -> list[str]:
    """Holdings in a stated order, so a test can vary only that."""
    return [*order, "S018", "S019"]


def test_the_pick_does_not_follow_the_order_of_the_holdings() -> None:
    """Same holdings, opposite iteration order, same sell."""
    forward, _ = topk_dropout_orders(SCORED, _held(TIED), topk=5, n_drop=1)
    reverse, _ = topk_dropout_orders(
        SCORED, _held(list(reversed(TIED))), topk=5, n_drop=1
    )
    assert forward == reverse, (
        f"the pick followed the input order: {forward} vs {reverse}"
    )


def test_the_tie_break_is_stated_not_inherited() -> None:
    """Among symbols tied at -inf, the lowest symbol is sold.

    Pinning the value, not just its stability, is what makes a replay
    reproduce a recorded book rather than merely agree with itself.
    """
    sells, _ = topk_dropout_orders(SCORED, _held(TIED), topk=5, n_drop=1)
    assert sells == [min(TIED)], f"expected {[min(TIED)]}, got {sells}"


def test_a_real_score_still_beats_an_absent_one() -> None:
    """The tie-break must not outrank the score it breaks ties within."""
    held = ["SZ301029", "S019"]
    sells, _ = topk_dropout_orders(SCORED, held, topk=5, n_drop=1)
    assert sells == ["SZ301029"], (
        f"an unscored holding should go before a scored one, got {sells}"
    )


def test_several_absent_holdings_are_dropped_in_a_stated_order() -> None:
    """n_drop > 1 has to pick a prefix, not an arbitrary subset."""
    held = ["SZ301293", "SZ301029", "SZ300999"]
    sells, _ = topk_dropout_orders(SCORED, held, topk=5, n_drop=2)
    assert sells == ["SZ300999", "SZ301029"], f"got {sells}"


def test_the_pick_survives_a_change_of_hash_seed() -> None:
    """End to end on a real set, which is what the pipeline passes.

    A list cannot reproduce the production shape; only a set can, and its
    iteration order is fixed per process by PYTHONHASHSEED.
    """
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from ashare_lab.paper.signal import topk_dropout_orders\n"
        f"scored = {SCORED!r}\n"
        f"held = set({_held(TIED)!r})\n"
        "sells, _ = topk_dropout_orders(scored, held, topk=5, n_drop=1)\n"
        "print(json.dumps(sells))\n"
    )
    picks = set()
    for seed in ("1", "2", "3", "4", "5", "6"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, (
            f"seed {seed} failed: {result.stderr.strip()[-400:]}"
        )
        picks.add(tuple(json.loads(result.stdout.strip().splitlines()[-1])))
    assert len(picks) == 1, f"the pick varies with the hash seed: {picks}"
    assert picks == {(min(TIED),)}, f"unexpected pick: {picks}"


def test_scored_ties_are_also_broken(
) -> None:
    """Two holdings on the same real score tie just as absent ones do."""
    scored = dict(SCORED)
    scored["SZ301029"] = -0.5
    scored["SZ301293"] = -0.5
    forward, _ = topk_dropout_orders(scored, _held(TIED), topk=5, n_drop=1)
    reverse, _ = topk_dropout_orders(
        scored, _held(list(reversed(TIED))), topk=5, n_drop=1
    )
    assert forward == reverse == [min(TIED)], f"{forward} vs {reverse}"

def test_the_buy_order_does_not_follow_the_signal_dict_order() -> None:
    """Two symbols on one score must rank the same either way round.

    ranked feeds the buy list, and a stable sort on score alone leaves
    tied symbols in the order the signals dict was built -- which is the
    order the prediction file happened to be read in.
    """
    tied_pair = ["AA999", "AB999"]
    base = {k: v for k, v in SCORED.items() if k not in tied_pair}
    forward = {**base, tied_pair[0]: 0.5, tied_pair[1]: 0.5}
    reverse = {**base, tied_pair[1]: 0.5, tied_pair[0]: 0.5}

    _, buys_f = topk_dropout_orders(forward, [], topk=5, n_drop=1)
    _, buys_r = topk_dropout_orders(reverse, [], topk=5, n_drop=1)
    assert buys_f == buys_r, (
        f"the buy order followed the dict order: {buys_f} vs {buys_r}"
    )


def test_tied_buys_rank_by_symbol() -> None:
    """The stated tie-break, so a replay reproduces a recorded buy list."""
    tied_pair = ["AB999", "AA999"]
    scored = {"S000": 9.0, tied_pair[0]: 0.5, tied_pair[1]: 0.5}
    _, buys = topk_dropout_orders(scored, [], topk=3, n_drop=0)
    assert buys == ["S000", "AA999", "AB999"], f"got {buys}"
