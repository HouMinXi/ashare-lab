"""Unit tests for ashare_lab.research.analyze_matrix.

Covers correlation (MultiIndex parquets), oracle (survivors-only,
window-intersected), kill gates (IC + MaxDD), JSONL schema validation,
and markdown output.  All tests run without qlib/torch.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ashare_lab.research.analyze_matrix import (
    compute_correlation_matrix,
    compute_oracle,
    generate_gate_decision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pred_parquet(path, dates, instruments, scores):
    """Write a prediction parquet with MultiIndex (datetime, instrument)."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), inst) for d, inst in zip(dates, instruments)],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"score": scores}, index=idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _make_jsonl(records):
    """Serialize records to JSONL string."""
    return "\n".join(json.dumps(r) for r in records)


def _base_record(model, window, ic=0.05, excess=0.03, maxdd=-0.05):
    """Create a valid JSONL record."""
    return {
        "model": model,
        "window": window,
        "ic": ic,
        "excess": excess,
        "maxdd": maxdd,
        "completed_at": "2026-07-08T14:30:00Z",
    }


# ---------------------------------------------------------------------------
# Correlation matrix tests
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_symmetric_and_diagonal(self, tmp_path):
        """Correlation matrix is symmetric with diagonal = 1.0."""
        dates = ["2026-01-01"] * 40
        instruments = [f"SH{i:06d}" for i in range(40)]
        # Candidate A: scores 0..39
        scores_a = list(range(40))
        # Candidate B: same rank order (perfect correlation)
        scores_b = [s * 2 for s in scores_a]

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _write_pred_parquet(dir_a / "pred_w1.parquet", dates, instruments, scores_a)
        _write_pred_parquet(dir_b / "pred_w1.parquet", dates, instruments, scores_b)

        corr = compute_correlation_matrix({"a": dir_a, "b": dir_b}, window_id=1)

        assert corr.loc["a", "a"] == pytest.approx(1.0)
        assert corr.loc["b", "b"] == pytest.approx(1.0)
        # Symmetric
        assert corr.loc["a", "b"] == pytest.approx(corr.loc["b", "a"])
        # Perfect rank correlation
        assert corr.loc["a", "b"] == pytest.approx(1.0, abs=0.01)

    def test_insufficient_common_instruments(self, tmp_path):
        """Off-diagonal is NaN when < 30 common instruments."""
        dates = ["2026-01-01"] * 20  # Only 20 instruments
        instruments = [f"SH{i:06d}" for i in range(20)]
        scores = list(range(20))

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _write_pred_parquet(dir_a / "pred_w1.parquet", dates, instruments, scores)
        _write_pred_parquet(
            dir_b / "pred_w1.parquet", dates, instruments, list(reversed(scores))
        )

        corr = compute_correlation_matrix({"a": dir_a, "b": dir_b}, window_id=1)

        assert corr.loc["a", "a"] == pytest.approx(1.0)
        assert pd.isna(corr.loc["a", "b"])


# ---------------------------------------------------------------------------
# Oracle tests
# ---------------------------------------------------------------------------


class TestOracle:
    def test_basic(self):
        """Oracle picks best-per-window, computes headroom correctly."""
        records = [
            _base_record("cand_a", 1, excess=0.10),
            _base_record("cand_a", 2, excess=0.05),
            _base_record("cand_a", 3, excess=0.08),
            _base_record("cand_b", 1, excess=0.03),
            _base_record("cand_b", 2, excess=0.12),
            _base_record("cand_b", 3, excess=0.02),
            _base_record("cand_c", 1, excess=0.01),
            _base_record("cand_c", 2, excess=0.04),
            _base_record("cand_c", 3, excess=0.15),
        ]
        incumbent = {1: 0.06, 2: 0.07, 3: 0.09}

        result = compute_oracle(records, incumbent)

        # Best per window: w1=cand_a(0.10), w2=cand_b(0.12), w3=cand_c(0.15)
        assert result["oracle_total"] == pytest.approx(0.10 + 0.12 + 0.15)
        assert result["incumbent_total"] == pytest.approx(0.06 + 0.07 + 0.09)
        assert result["headroom"] == pytest.approx(
            result["oracle_total"] - result["incumbent_total"]
        )
        assert result["n_common_windows"] == 3
        assert len(result["winner_matrix"]) == 3
        winners = {w["window"]: w["winner"] for w in result["winner_matrix"]}
        assert winners[1] == "cand_a"
        assert winners[2] == "cand_b"
        assert winners[3] == "cand_c"

    def test_incumbent_wins_all(self):
        """When incumbent is best everywhere, headroom = 0."""
        records = [
            _base_record("cand_a", 1, excess=0.01),
            _base_record("cand_a", 2, excess=0.02),
        ]
        incumbent = {1: 0.10, 2: 0.20}

        result = compute_oracle(records, incumbent)

        assert result["headroom"] == pytest.approx(0.0)
        winners = {w["window"]: w["winner"] for w in result["winner_matrix"]}
        assert all(w == "incumbent" for w in winners.values())

    def test_window_intersection(self):
        """Oracle only compares common windows (N11 fix)."""
        # Records for windows 1-5 only.
        records = [_base_record("cand_a", w, excess=0.05) for w in range(1, 6)]
        # Incumbent has windows 1-10.
        incumbent = {w: 0.03 for w in range(1, 11)}

        result = compute_oracle(records, incumbent)

        # Only 5 common windows, not 10.
        assert result["n_common_windows"] == 5
        assert len(result["winner_matrix"]) == 5
        # Incumbent total only over the 5 common windows.
        assert result["incumbent_total"] == pytest.approx(0.03 * 5)


# ---------------------------------------------------------------------------
# Gate decision tests
# ---------------------------------------------------------------------------


class TestGateDecision:
    def test_oracle_excludes_dead(self, tmp_path):
        """Killed candidates do NOT appear in oracle winner_matrix (N3 fix)."""
        # Candidate A: alive, high excess.
        # Candidate B: killed by IC < 0.02.
        records = [
            _base_record("a_alive", 1, ic=0.05, excess=0.10),
            _base_record("a_alive", 2, ic=0.04, excess=0.08),
            _base_record("b_dead", 1, ic=0.01, excess=0.50),  # IC kills it
            _base_record("b_dead", 2, ic=0.01, excess=0.40),
        ]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(_make_jsonl(records))

        incumbent = {1: 0.05, 2: 0.05}

        path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess=incumbent,
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )

        md_content = path.read_text()
        # b_dead should be DEAD, not in oracle winners.
        assert "b_dead" in md_content
        assert "DEAD" in md_content
        # Oracle winner_matrix should only have a_alive or incumbent.
        # b_dead had excess=0.50 which would dominate -- proving exclusion works.
        assert "| 1 | a_alive |" in md_content  # a_alive(0.10) > incumbent(0.05)

    def test_kills_low_ic(self, tmp_path):
        """Candidate with IC < 0.02 in any window is marked DEAD."""
        records = [
            _base_record("good", 1, ic=0.05, excess=0.03),
            _base_record("bad_ic", 1, ic=0.01, excess=0.03),
        ]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(_make_jsonl(records))

        path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )

        md = path.read_text()
        # bad_ic should be DEAD with IC reason.
        assert "bad_ic" in md
        assert "DEAD" in md
        assert "IC" in md

    def test_kills_high_maxdd(self, tmp_path):
        """Candidate with maxdd worse than -0.30 is marked DEAD (MAT-06)."""
        records = [
            _base_record("good", 1, ic=0.05, excess=0.03, maxdd=-0.10),
            _base_record("bad_dd", 1, ic=0.05, excess=0.03, maxdd=-0.35),
        ]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(_make_jsonl(records))

        path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )

        md = path.read_text()
        assert "bad_dd" in md
        assert "DEAD" in md
        assert "MaxDD" in md

    def test_md_output_sections(self, tmp_path):
        """GATE_DECISION.md contains expected sections."""
        records = [
            _base_record("cand_a", 1),
            _base_record("cand_a", 2),
        ]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(_make_jsonl(records))

        path = generate_gate_decision(
            results_jsonl=jsonl_path,
            incumbent_excess={1: 0.02, 2: 0.02},
            kill_ic=0.02,
            kill_maxdd=-0.30,
            output_dir=tmp_path,
        )

        assert path.name == "GATE_DECISION.md"
        md = path.read_text()
        assert "## Candidate Verdicts" in md
        assert "## Oracle Analysis" in md
        assert "## Recommendations" in md
        assert "D-14" in md  # Oracle framing warning

    def test_jsonl_schema_validation(self, tmp_path):
        """Missing required key in JSONL record raises ValueError."""
        # Record missing "maxdd" key.
        bad_record = {
            "model": "test",
            "window": 1,
            "ic": 0.05,
            "excess": 0.03,
            "completed_at": "2026-07-08T14:30:00Z",
        }
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(json.dumps(bad_record))

        with pytest.raises(ValueError, match="maxdd"):
            generate_gate_decision(
                results_jsonl=jsonl_path,
                incumbent_excess={1: 0.02},
                kill_ic=0.02,
                kill_maxdd=-0.30,
                output_dir=tmp_path,
            )
