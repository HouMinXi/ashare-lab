"""Post-hoc analysis of model matrix experiment results.

Reads JSONL results and prediction parquets produced by batch_experiment.py.
Computes Spearman correlation matrices, per-window winner/oracle headroom,
and generates GATE_DECISION.md with per-candidate verdicts.

Pure reader -- no qlib, no torch, no GPU.  stdlib + pandas + scipy only.
Runs on X500 after batch_experiment.py completes on GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

# Shared JSONL record schema -- single source of truth in metrics.py.
from ashare_lab.research.metrics import CELL_SCHEMA_KEYS

# Minimum common instruments per date to compute meaningful Spearman rho.
_MIN_COMMON_INSTRUMENTS = 30


def compute_correlation_matrix(
    result_dirs: dict[str, Path],
    window_id: int,
) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlation between candidate predictions.

    Args:
        result_dirs: Mapping of candidate tag to its predictions directory.
            Each dir should contain pred_w{window_id}.parquet files.
        window_id: Which walk-forward window to correlate.

    Returns:
        Symmetric DataFrame (tags x tags), diagonal=1.0, off-diagonal=Spearman
        rho.  NaN when fewer than _MIN_COMMON_INSTRUMENTS common instruments.
    """
    # Load predictions: MultiIndex (datetime, instrument) + "score" column.
    preds: dict[str, pd.Series] = {}
    for tag, pred_dir in result_dirs.items():
        pf = Path(pred_dir) / f"pred_w{window_id}.parquet"
        if not pf.exists():
            log.warning("missing parquet for %s window %d: %s", tag, window_id, pf)
            continue
        df = pd.read_parquet(pf)
        preds[tag] = df["score"]  # Series with MultiIndex (datetime, instrument)

    tags = sorted(preds.keys())
    n = len(tags)
    corr = pd.DataFrame(float("nan"), index=tags, columns=tags)
    # Diagonal = 1.0
    for t in tags:
        corr.loc[t, t] = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = preds[tags[i]], preds[tags[j]]
            # Align by date, then by instrument within each date.
            dates_i = si.index.get_level_values(0).unique()
            dates_j = sj.index.get_level_values(0).unique()
            common_dates = dates_i.intersection(dates_j)

            all_si, all_sj = [], []
            for date in common_dates:
                pi = si.xs(date, level=0).dropna()
                pj = sj.xs(date, level=0).dropna()
                common_instr = pi.index.intersection(pj.index)
                if len(common_instr) >= _MIN_COMMON_INSTRUMENTS:
                    all_si.extend(pi.loc[common_instr].values)
                    all_sj.extend(pj.loc[common_instr].values)

            if len(all_si) >= _MIN_COMMON_INSTRUMENTS:
                rho, _ = spearmanr(all_si, all_sj)
                corr.loc[tags[i], tags[j]] = rho
                corr.loc[tags[j], tags[i]] = rho

    return corr


def compute_oracle(
    records: list[dict],
    incumbent_excess: dict[int, float],
) -> dict:
    """Compute oracle (best-per-window) headroom over incumbent.

    Only receives survivor records -- caller must filter out killed candidates
    before calling (dead candidates inflate oracle headroom).

    Uses window intersection: only compares windows present in BOTH
    the records and incumbent_excess.

    Args:
        records: List of JSONL record dicts for surviving candidates only.
        incumbent_excess: {window_id: excess_return} for incumbent blend.

    Returns:
        Dict with oracle_total, incumbent_total, headroom, n_common_windows,
        winner_matrix.
    """
    # Group records by window.
    by_window: dict[int, list[dict]] = {}
    for r in records:
        by_window.setdefault(r["window"], []).append(r)

    # Intersect windows: oracle must not credit windows the incumbent never traded.
    record_windows = set(by_window.keys())
    incumbent_windows = set(incumbent_excess.keys())
    common_windows = sorted(record_windows & incumbent_windows)

    oracle_total = 0.0
    incumbent_total = 0.0
    winner_matrix: list[dict] = []

    for wid in common_windows:
        candidates = by_window.get(wid, [])
        # Include incumbent in the pool.
        all_excess = [(c["model"], c["excess"]) for c in candidates]
        inc_val = incumbent_excess[wid]
        all_excess.append(("incumbent", inc_val))

        best_tag, best_excess = max(all_excess, key=lambda x: x[1])
        oracle_total += best_excess
        incumbent_total += inc_val
        winner_matrix.append({
            "window": wid,
            "winner": best_tag,
            "excess": best_excess,
        })

    return {
        "oracle_total": oracle_total,
        "incumbent_total": incumbent_total,
        "headroom": oracle_total - incumbent_total,
        "n_common_windows": len(common_windows),
        "winner_matrix": winner_matrix,
    }


def generate_gate_decision(
    results_jsonl: Path,
    incumbent_excess: dict[int, float],
    kill_ic: float,
    kill_maxdd: float,
    output_dir: Path,
    result_dirs: dict[str, Path] | None = None,
) -> Path:
    """Generate GATE_DECISION.md from matrix experiment results.

    Applies the kill gate (IC < kill_ic OR maxdd worse than kill_maxdd)
    per candidate.  Kill gate runs FIRST, then only survivors go
    to oracle computation.

    Args:
        results_jsonl: Path to results.jsonl with all candidate results.
        incumbent_excess: {window_id: excess_return} for incumbent blend.
        kill_ic: IC threshold -- candidate killed if any window IC < this.
        kill_maxdd: MaxDD threshold (negative) -- candidate killed if any
            window maxdd worse (more negative) than this.
        output_dir: Directory for GATE_DECISION.md output.
        result_dirs: Optional dict mapping candidate tag to predictions dir.
            When provided, computes correlation matrix among survivors.

    Returns:
        Path to GATE_DECISION.md.
    """
    # Read and validate JSONL records.
    records: list[dict] = []
    for line in results_jsonl.read_text().strip().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [k for k in CELL_SCHEMA_KEYS if k not in rec]
        if missing:
            raise ValueError(
                f"JSONL record missing required keys {missing}: {rec}"
            )
        records.append(rec)

    # Group by model (candidate tag).
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    # Apply kill gate per candidate.
    verdicts: list[dict] = []
    survivors: list[dict] = []

    for tag in sorted(by_model.keys()):
        candidate_records = by_model[tag]
        n_windows = len(candidate_records)
        ics = [r["ic"] for r in candidate_records]
        excesses = [r["excess"] for r in candidate_records]
        maxdds = [r["maxdd"] for r in candidate_records]

        mean_ic = sum(ics) / len(ics) if ics else 0.0
        total_excess = sum(excesses)
        worst_maxdd = min(maxdds) if maxdds else 0.0

        # Kill check: any single window below threshold kills the candidate.
        kill_reason = None
        for r in candidate_records:
            if r["ic"] < kill_ic:
                kill_reason = f"IC {r['ic']:.4f} < {kill_ic} in window {r['window']}"
                break
            if r["maxdd"] < kill_maxdd:
                kill_reason = (
                    f"MaxDD {r['maxdd']:.4f} < {kill_maxdd} in window {r['window']}"
                )
                break

        status = "DEAD" if kill_reason else "ALIVE"
        verdicts.append({
            "tag": tag,
            "status": status,
            "n_windows": n_windows,
            "mean_ic": mean_ic,
            "total_excess": total_excess,
            "worst_maxdd": worst_maxdd,
            "kill_reason": kill_reason or "--",
        })

        if status == "ALIVE":
            survivors.extend(candidate_records)

    # Oracle only sees survivors: dead candidates would inflate headroom.
    oracle = compute_oracle(survivors, incumbent_excess)

    # Correlation among survivors (if result_dirs provided).
    low_corr_pairs: list[tuple[str, str, float]] = []
    if result_dirs:
        survivor_tags = {v["tag"] for v in verdicts if v["status"] == "ALIVE"}
        survivor_dirs = {
            t: d for t, d in result_dirs.items() if t in survivor_tags
        }
        if len(survivor_dirs) >= 2 and oracle["n_common_windows"] > 0:
            # Use first common window for correlation snapshot. Single-window
            # Spearman rho is a directional indicator, not a definitive measure;
            # full multi-window correlation is deferred to manual analysis.
            first_window = oracle["winner_matrix"][0]["window"]
            corr_df = compute_correlation_matrix(survivor_dirs, first_window)
            tags = list(corr_df.index)
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    val = corr_df.loc[tags[i], tags[j]]
                    if pd.notna(val) and val < 0.6:
                        low_corr_pairs.append((tags[i], tags[j], float(val)))

    # Write GATE_DECISION.md
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "GATE_DECISION.md"

    lines = ["# Gate Decision Report\n"]

    # Section 1: Candidate Verdicts
    lines.append("## Candidate Verdicts\n")
    lines.append(
        "| Candidate | Status | Windows | Mean IC | Total Excess "
        "| Worst MaxDD | Kill Reason |"
    )
    lines.append(
        "|-----------|--------|---------|---------|----------"
        "---|-------------|-------------|"
    )
    for v in verdicts:
        lines.append(
            f"| {v['tag']} | {v['status']} | {v['n_windows']} "
            f"| {v['mean_ic']:.4f} | {v['total_excess']:.4f} "
            f"| {v['worst_maxdd']:.4f} | {v['kill_reason']} |"
        )
    lines.append("")

    # Section 2: Oracle Analysis
    lines.append("## Oracle Analysis\n")
    lines.append(
        f"- **Oracle total** (best-per-window sum): {oracle['oracle_total']:.4f}"
    )
    lines.append(f"- **Incumbent total**: {oracle['incumbent_total']:.4f}")
    lines.append(f"- **Headroom**: {oracle['headroom']:.4f}")
    lines.append(f"- **Common windows**: {oracle['n_common_windows']}")
    lines.append("")
    # Oracle headroom framing: kill-only, never a promise.
    lines.append(
        "> Oracle headroom is used to KILL combination work when headroom is "
        "tiny, never to promise gains."
    )
    lines.append("")

    if oracle["winner_matrix"]:
        lines.append("### Per-Window Winners\n")
        lines.append("| Window | Winner | Excess |")
        lines.append("|--------|--------|--------|")
        for w in oracle["winner_matrix"]:
            lines.append(f"| {w['window']} | {w['winner']} | {w['excess']:.4f} |")
        lines.append("")

    # Section 3: Low-Correlation Pairs (if result_dirs provided)
    if result_dirs:
        lines.append("## Low-Correlation Pairs\n")
        if low_corr_pairs:
            lines.append(
                "Pairs with Spearman rho < 0.6 (candidates for Phase 2 "
                "combination testing when correlation < 0.6):\n"
            )
            lines.append("| Pair A | Pair B | Spearman rho |")
            lines.append("|--------|--------|--------------|")
            for a, b, rho in low_corr_pairs:
                lines.append(f"| {a} | {b} | {rho:.4f} |")
        else:
            lines.append("No low-correlation pairs found among survivors.")
        lines.append("")

    # Section 4: Recommendations
    # Ranking uses after-cost excess + bear floor, not IC alone.
    lines.append("## Recommendations\n")
    alive = [v for v in verdicts if v["status"] == "ALIVE"]
    if alive:
        # Rank survivors by total excess.
        alive_sorted = sorted(alive, key=lambda v: v["total_excess"], reverse=True)
        lines.append(
            "Survivors ranked by total after-cost excess return ("
            "IC is kill floor only, ranking uses excess + bear floor):\n"
        )
        for rank, v in enumerate(alive_sorted, 1):
            lines.append(
                f"{rank}. **{v['tag']}**: excess={v['total_excess']:.4f}, "
                f"IC={v['mean_ic']:.4f}, worst_maxdd={v['worst_maxdd']:.4f}"
            )
    else:
        lines.append("All candidates killed. No survivors for combination testing.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("GATE_DECISION.md written: %s", md_path)
    return md_path


def main() -> None:
    """CLI entry point for analyze_matrix.py."""
    parser = argparse.ArgumentParser(
        description="Analyze model matrix experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("matrix_results"),
        help="Directory containing results.jsonl and per-candidate subdirs",
    )
    parser.add_argument(
        "--incumbent-json",
        type=Path,
        required=True,
        help="JSON file with incumbent per-window excess returns {window_id: excess}",
    )
    parser.add_argument(
        "--kill-ic",
        type=float,
        default=0.02,
        help="IC kill threshold (default: 0.02)",
    )
    parser.add_argument(
        "--kill-maxdd",
        type=float,
        default=-0.30,
        help="MaxDD kill threshold (default: -0.30)",
    )
    args = parser.parse_args()

    # Load incumbent excess.
    incumbent_raw = json.loads(args.incumbent_json.read_text())
    # Keys may be strings from JSON; convert to int.
    incumbent_excess = {int(k): float(v) for k, v in incumbent_raw.items()}

    # Discover result_dirs: each subdirectory = candidate tag.
    results_jsonl = args.results_dir / "results.jsonl"
    result_dirs: dict[str, Path] = {}
    for d in sorted(args.results_dir.iterdir()):
        if d.is_dir():
            result_dirs[d.name] = d

    gate_path = generate_gate_decision(
        results_jsonl=results_jsonl,
        incumbent_excess=incumbent_excess,
        kill_ic=args.kill_ic,
        kill_maxdd=args.kill_maxdd,
        output_dir=args.results_dir,
        result_dirs=result_dirs if result_dirs else None,
    )
    print(f"Gate decision written: {gate_path}")


if __name__ == "__main__":
    main()
