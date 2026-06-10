# ashare-lab

A-share daily-frequency paper-trading lab: Qlib + LightGBM research, a local
fill-simulation ledger (T+1, price limits, lots, fees), and WeChat daily reports
delivered via hermes.

Paper trading only — no real orders. The live-trading gate (broker programmatic
trading report + QMT) is documented in the plan and explicitly out of scope.

## Layout

- `ashare_lab/` — package: data pipeline, research, paper engine, report, sentiment
- `configs/` — strategy and pipeline configuration
- `scripts/` — daily pipeline, deploy, systemd units
- `tests/` — rule-level tests for the paper engine

## Development

All work happens in the `.worktrees/dev` linked worktree, never in the main tree.

```bash
cd .worktrees/dev
uv sync --all-groups
uv run python -c "import qlib"
```

Runtime host: always-on Linux mini-PC; daily pipeline fires at 01:00 Asia/Shanghai
via systemd timer, after the upstream data release (~23:30 CST).
