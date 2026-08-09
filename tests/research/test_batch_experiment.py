"""Tests for batch_experiment.py orchestrator.

All tests mock subprocess.run -- no real SSH/SCP calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import importlib.util

import pytest

from ashare_lab.research.batch_experiment import (
    _DEFAULT_GPU_IP,
    _kill_remote_python,
    _recover_result_from_gpu,
    append_result,
    commit_summary_csv,
    discover_configs,
    load_completed,
    run_matrix,
    scp_pred_from_gpu,
    scp_results,
)
from ashare_lab.research.metrics import CELL_SCHEMA_KEYS


@pytest.fixture(autouse=True)
def _mock_gpu_online(monkeypatch):
    """Prevent real WOL/ping in all tests."""
    monkeypatch.setattr(
        "ashare_lab.research.batch_experiment.ensure_gpu_online", lambda *a, **kw: None,
    )


def _make_record(model: str = "a_lgb", window: int = 1, **overrides) -> dict:
    """Build a minimal valid CELL_SCHEMA record."""
    rec = {
        "model": model,
        "window": window,
        "ic": 0.05,
        "excess": 0.02,
        "maxdd": -0.10,
        "completed_at": "2026-07-01T00:00:00+00:00",
    }
    rec.update(overrides)
    return rec


def _write_matrix_yaml(d: Path, name: str, tag: str, model_type: str = "lgbm",
                        seeds: list[int] | None = None) -> Path:
    """Write a minimal matrix_*.yaml candidate config."""
    content = {
        "model": {"type": model_type, "tag": tag},
        "matrix": {"tag": tag},
    }
    if seeds is not None:
        content["seeds"] = seeds
    p = d / name
    p.write_text(
        "---\n"
        + "\n".join(f"{k}: {json.dumps(v)}" for k, v in content.items())
        + "\n",
    )
    # Re-write as proper yaml for safe_load compatibility.
    import yaml
    p.write_text(yaml.safe_dump(content))
    return p


# -- test_discover_configs --

def test_discover_configs(tmp_path: Path) -> None:
    _write_matrix_yaml(tmp_path, "matrix_b.yaml", "b")
    _write_matrix_yaml(tmp_path, "matrix_a.yaml", "a")
    _write_matrix_yaml(tmp_path, "matrix_c.yaml", "c")
    (tmp_path / "baseline.yaml").write_text("---\n")

    result = discover_configs(str(tmp_path))
    names = [p.name for p in result]
    assert names == ["matrix_a.yaml", "matrix_b.yaml", "matrix_c.yaml"]


# -- test_load_completed_empty --

def test_load_completed_empty(tmp_path: Path) -> None:
    assert load_completed(str(tmp_path / "nope.jsonl")) == set()


# -- test_load_completed_parses --

def test_load_completed_parses(tmp_path: Path) -> None:
    jf = tmp_path / "results.jsonl"
    jf.write_text(
        json.dumps(_make_record("x", 1)) + "\n"
        + json.dumps(_make_record("y", 2)) + "\n"
    )
    result = load_completed(str(jf))
    assert result == {("x", 1), ("y", 2)}


# -- test_resume_skips_completed --

@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_resume_skips_completed(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(configs_dir, "matrix_a.yaml", "a_lgb")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl = out_dir / "results.jsonl"
    # Pre-populate: window 1 already done.
    jsonl.write_text(json.dumps(_make_record("a_lgb", 1)) + "\n")

    with patch("ashare_lab.research.batch_experiment.load_completed") as mock_lc:
        mock_lc.return_value = {("a_lgb", 1)}
        run_matrix(
            configs_dir=str(configs_dir),
            output_dir=str(out_dir),
            windows=[1],
            gpu_host="fake",
            gpu_repo="/fake",
        )

    # subprocess.run should NOT have been called for SSH (only scp calls).
    for c in mock_run.call_args_list:
        args = c[0][0] if c[0] else c[1].get("args", "")
        if isinstance(args, str):
            assert "matrix_runner" not in args


# -- test_smoke_mode --

@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_smoke_mode(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(configs_dir, "matrix_a.yaml", "a")
    _write_matrix_yaml(configs_dir, "matrix_b.yaml", "b")
    _write_matrix_yaml(configs_dir, "matrix_c.yaml", "c")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    record = _make_record("a", 1)
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(record) + "\n",
        stderr="",
    )

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        smoke=True,
        windows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        gpu_host="fake",
        gpu_repo="/fake",
    )

    # Exactly 1 SSH call (first config x first window).
    ssh_calls = [
        c for c in mock_run.call_args_list
        if isinstance(c[0][0] if c[0] else "", str)
        and "matrix_runner" in str(c[0][0] if c[0] else "")
    ]
    assert len(ssh_calls) == 1


# -- test_smoke_passes_n_epochs --

@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_smoke_passes_n_epochs(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(configs_dir, "matrix_a.yaml", "a")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    record = _make_record("a", 1)
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(record) + "\n",
        stderr="",
    )

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        smoke=True,
        windows=[1],
        gpu_host="fake",
        gpu_repo="/fake",
    )

    ssh_calls = [
        c for c in mock_run.call_args_list
        if isinstance(c[0][0] if c[0] else "", str)
        and "matrix_runner" in str(c[0][0] if c[0] else "")
    ]
    assert len(ssh_calls) == 1
    cmd = ssh_calls[0][0][0]
    assert "--n-epochs" in cmd


# -- test_append_result --

def test_append_result(tmp_path: Path) -> None:
    jf = tmp_path / "results.jsonl"
    append_result(str(jf), _make_record("a", 1))
    append_result(str(jf), _make_record("b", 2))

    lines = jf.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        for k in CELL_SCHEMA_KEYS:
            assert k in rec


# -- test_commit_summary_csv_project_root --

@patch("subprocess.run")
def test_commit_summary_csv_project_root(
    mock_run: MagicMock, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proj_root = tmp_path / "proj"
    proj_root.mkdir()

    jsonl = out_dir / "results.jsonl"
    jsonl.write_text(json.dumps(_make_record("a", 1)) + "\n")

    mock_run.return_value = MagicMock(returncode=0)

    commit_summary_csv(str(out_dir), str(proj_root))

    # CSV at project_root, not output_dir.
    assert (proj_root / "matrix_summary.csv").exists()
    assert not (out_dir / "matrix_summary.csv").exists()

    # git commands use cwd=project_root.
    for c in mock_run.call_args_list:
        assert c[1].get("cwd") == str(proj_root)


# -- test_scp_not_rsync --

@patch("subprocess.run", return_value=MagicMock(returncode=0))
def test_scp_not_rsync(mock_run: MagicMock) -> None:
    scp_pred_from_gpu("host", "/repo", "subdir", "/tmp/local", "f.parquet")
    scp_results("/tmp/out", "tag", "host", "/repo")

    for c in mock_run.call_args_list:
        cmd = c[0][0]
        assert "scp" in cmd
        if isinstance(cmd, list):
            assert "rsync" not in cmd


# -- test_gpu_lock_context_manager --

@patch("fcntl.flock")
def test_gpu_lock_context_manager(
    mock_flock: MagicMock,
    tmp_path: Path,
) -> None:
    """Lock fd is released even when SSH raises."""
    from ashare_lab.research.batch_experiment import _gpu_lock

    with pytest.raises(RuntimeError, match="boom"):
        with _gpu_lock():
            raise RuntimeError("boom")

    # flock called twice: LOCK_EX + LOCK_UN.
    assert mock_flock.call_count == 2


# -- test_multi_seed_config --

@patch("ashare_lab.research.batch_experiment._average_seeds")
@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_multi_seed_config(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    mock_avg: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(
        configs_dir, "matrix_a.yaml", "a", seeds=[42, 123],
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_run(cmd, **kwargs):
        return MagicMock(
            returncode=0,
            stdout=json.dumps(_make_record("a_seed42", 1)) + "\n",
            stderr="",
        )

    mock_run.side_effect = fake_run

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        windows=[1],
        gpu_host="fake",
        gpu_repo="/fake",
    )

    # 2 SSH calls (one per seed).
    ssh_calls = [
        c for c in mock_run.call_args_list
        if isinstance(c[0][0] if c[0] else "", str)
        and "matrix_runner" in str(c[0][0] if c[0] else "")
    ]
    assert len(ssh_calls) == 2

    # _average_seeds called once.
    assert mock_avg.call_count == 1


# -- test_multi_seed_resume_no_false_skip --

@patch("ashare_lab.research.batch_experiment._average_seeds")
@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_multi_seed_resume_no_false_skip(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    mock_avg: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(
        configs_dir, "matrix_a.yaml", "a", seeds=[42, 123],
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl = out_dir / "results.jsonl"
    # Only seed 42 is done; seed 123 must still run.
    jsonl.write_text(json.dumps(_make_record("a_seed42", 1)) + "\n")

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(_make_record("a_seed123", 1)) + "\n",
        stderr="",
    )

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        windows=[1],
        gpu_host="fake",
        gpu_repo="/fake",
    )

    # Exactly 1 SSH call (seed 123 only; seed 42 skipped).
    ssh_calls = [
        c for c in mock_run.call_args_list
        if isinstance(c[0][0] if c[0] else "", str)
        and "matrix_runner" in str(c[0][0] if c[0] else "")
    ]
    assert len(ssh_calls) == 1
    assert "seed 123" in ssh_calls[0][0][0] or "--seed 123" in ssh_calls[0][0][0]


# -- test_ssh_uses_relative_paths --

@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_ssh_uses_relative_paths(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(configs_dir, "matrix_a.yaml", "a")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(_make_record("a", 1)) + "\n",
        stderr="",
    )

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        windows=[1],
        gpu_host="fake",
        gpu_repo="/fake",
    )

    ssh_calls = [
        c for c in mock_run.call_args_list
        if isinstance(c[0][0] if c[0] else "", str)
        and "matrix_runner" in str(c[0][0] if c[0] else "")
    ]
    assert len(ssh_calls) == 1
    cmd = ssh_calls[0][0][0]
    # Must use "configs/{name}" not "/home/..." absolute paths.
    assert "configs/matrix_a.yaml" in cmd
    assert "/home/" not in cmd


# -- test_commit_csv_skipped_on_scp_failure --

@patch("ashare_lab.research.batch_experiment.commit_summary_csv")
@patch("ashare_lab.research.batch_experiment.scp_results", return_value=False)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_commit_csv_skipped_on_scp_failure(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    mock_commit: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(configs_dir, "matrix_a.yaml", "a")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(_make_record("a", 1)) + "\n",
        stderr="",
    )

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        windows=[1],
        gpu_host="fake",
        gpu_repo="/fake",
        project_root=str(tmp_path),
    )

    # scp_results returned False -> commit_summary_csv NOT called.
    mock_commit.assert_not_called()


# -- test_multi_seed_labels_scp --

@patch("ashare_lab.research.batch_experiment._average_seeds")
@patch("ashare_lab.research.batch_experiment.scp_results", return_value=True)
@patch("ashare_lab.research.batch_experiment.scp_pred_from_gpu")
@patch("ashare_lab.research.batch_experiment.scp_config_to_gpu")
@patch("subprocess.run")
def test_multi_seed_labels_scp(
    mock_run: MagicMock,
    mock_scp_cfg: MagicMock,
    mock_scp_pred: MagicMock,
    mock_scp_res: MagicMock,
    mock_avg: MagicMock,
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_matrix_yaml(
        configs_dir, "matrix_a.yaml", "a", seeds=[42, 123],
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(_make_record("a_seed42", 1)) + "\n",
        stderr="",
    )

    run_matrix(
        configs_dir=str(configs_dir),
        output_dir=str(out_dir),
        windows=[1],
        gpu_host="fake",
        gpu_repo="/fake",
    )

    # scp_pred_from_gpu(host, repo, remote_subdir, local_dir, filename)
    # filename is the 5th positional arg (index 4).
    filenames = [c[0][4] for c in mock_scp_pred.call_args_list]
    label_only = [f for f in filenames if f.startswith("labels_")]
    pred_only = [f for f in filenames if f.startswith("pred_")]
    assert len(label_only) == 1
    assert len(pred_only) == 2


# -- test_average_seeds_direct --

@patch("ashare_lab.research.backtest.run_backtest")
@patch("ashare_lab.research.smoke_test.get_window")
def test_average_seeds_direct(
    mock_gw: MagicMock,
    mock_bt: MagicMock,
    tmp_path: Path,
) -> None:
    """Exercise _average_seeds with real parquet files (not mocked)."""
    import pandas as pd

    from ashare_lab.research.batch_experiment import _average_seeds

    tag_dir = tmp_path / "a"
    tag_dir.mkdir()

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), "SH600000"),
         (pd.Timestamp("2021-07-01"), "SH600001")],
        names=["datetime", "instrument"],
    )
    # Seed 42: scores [0.1, 0.3]; seed 123: scores [0.3, 0.5].
    pd.DataFrame({"score": [0.1, 0.3]}, index=idx).to_parquet(
        tag_dir / "pred_w1_seed42.parquet"
    )
    pd.DataFrame({"score": [0.3, 0.5]}, index=idx).to_parquet(
        tag_dir / "pred_w1_seed123.parquet"
    )
    pd.DataFrame({"label": [0.02, -0.01]}, index=idx).to_parquet(
        tag_dir / "labels_w1.parquet"
    )

    mock_gw.return_value = {
        "window_id": 1, "test_start": "2021-07-01", "test_end": "2021-12-31",
    }
    mock_bt.return_value = (pd.DataFrame({"return": [0.01]}), pd.Series([100.0]), 0)

    jsonl = tmp_path / "results.jsonl"
    completed: set[tuple[str, int]] = set()

    with patch("ashare_lab.config.load_config") as mock_cfg:
        mock_cfg.return_value = {"strategy": {"main": {"n_drop": 1}}}
        _average_seeds(str(tmp_path), "a", 1, [42, 123], str(jsonl), completed)

    # Averaged parquet written.
    avg_path = tag_dir / "pred_w1.parquet"
    assert avg_path.exists()
    avg_df = pd.read_parquet(avg_path)
    assert abs(avg_df["score"].iloc[0] - 0.2) < 1e-9  # (0.1+0.3)/2
    assert abs(avg_df["score"].iloc[1] - 0.4) < 1e-9  # (0.3+0.5)/2

    # JSONL record written.
    assert ("a_avg", 1) in completed
    rec = json.loads(jsonl.read_text().strip())
    assert rec["model"] == "a_avg"


# -- test_kill_remote_python --

@patch("subprocess.run")
def test_kill_remote_python_success(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    _kill_remote_python("fake@host")
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "taskkill" in cmd
    assert "python.exe" in cmd


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 15))
def test_kill_remote_python_timeout(mock_run: MagicMock) -> None:
    """Timeout is swallowed, not raised."""
    _kill_remote_python("fake@host")  # must not raise


# -- test_recover_result_from_gpu --

@patch("subprocess.run")
def test_recover_result_success(mock_run: MagicMock, tmp_path: Path) -> None:
    record = _make_record("a", 1)
    local_path = tmp_path / "a" / "result_w1.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    def fake_scp(cmd, **kwargs):
        # Simulate SCP writing the file.
        local_path.write_text(json.dumps(record))
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_scp
    result = _recover_result_from_gpu("host", "/repo", "a", 1, str(tmp_path))
    assert result == record


@patch("subprocess.run")
def test_recover_result_scp_fails(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=1)
    result = _recover_result_from_gpu("host", "/repo", "a", 1, str(tmp_path))
    assert result is None


@patch("subprocess.run")
def test_recover_result_with_seed(mock_run: MagicMock, tmp_path: Path) -> None:
    record = _make_record("a_seed42", 1)
    local_path = tmp_path / "a" / "result_w1_seed42.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    def fake_scp(cmd, **kwargs):
        local_path.write_text(json.dumps(record))
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_scp
    result = _recover_result_from_gpu(
        "host", "/repo", "a", 1, str(tmp_path), seed=42,
    )
    assert result == record
    # Verify correct filename was used.
    scp_cmd = mock_run.call_args[0][0]
    assert "result_w1_seed42.json" in str(scp_cmd)


# -- ensure_gpu_online / _ping_ok unit tests --


def _real_ensure_gpu_online():
    """Import the real function, bypassing autouse mock."""
    import importlib
    import ashare_lab.research.batch_experiment as mod
    importlib.reload(mod)
    return mod.ensure_gpu_online


def test_ensure_gpu_already_online(monkeypatch: pytest.MonkeyPatch) -> None:
    import ashare_lab.research.batch_experiment as mod
    real_fn = _real_ensure_gpu_online()
    monkeypatch.setattr(mod, "_ping_ok", lambda ip: True)
    real_fn()  # should return immediately


@pytest.mark.skipif(importlib.util.find_spec("wakeonlan") is None, reason="wakeonlan not installed on this host")
def test_ensure_gpu_wol_then_online(monkeypatch: pytest.MonkeyPatch) -> None:
    import ashare_lab.research.batch_experiment as mod
    real_fn = _real_ensure_gpu_online()
    ping_results = iter([False, False, True])
    monkeypatch.setattr(mod, "_ping_ok", lambda ip: next(ping_results))
    monkeypatch.setattr(mod, "_wait_ssh", lambda ip, **kw: None)
    monkeypatch.setattr("wakeonlan.wake", lambda mac, **kw: None)
    real_fn(timeout=30, poll=0)


@pytest.mark.skipif(importlib.util.find_spec("wakeonlan") is None, reason="wakeonlan not installed on this host")
def test_ensure_gpu_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import ashare_lab.research.batch_experiment as mod
    real_fn = _real_ensure_gpu_online()
    monkeypatch.setattr(mod, "_ping_ok", lambda ip: False)
    monkeypatch.setattr("wakeonlan.wake", lambda mac, **kw: None)
    with pytest.raises(RuntimeError, match="did not respond"):
        real_fn(timeout=1, poll=0)


@patch("subprocess.run")
def test_ping_ok_success(mock_run: MagicMock) -> None:
    from ashare_lab.research.batch_experiment import _ping_ok
    mock_run.return_value = MagicMock(returncode=0)
    assert _ping_ok("1.2.3.4") is True
    cmd = mock_run.call_args[0][0]
    assert "ping" in cmd
    assert "1.2.3.4" in cmd


@patch("subprocess.run")
def test_ping_ok_failure(mock_run: MagicMock) -> None:
    from ashare_lab.research.batch_experiment import _ping_ok
    mock_run.return_value = MagicMock(returncode=1)
    assert _ping_ok("1.2.3.4") is False


# -- W4: ollama service control tests --


@patch("subprocess.run")
def test_gpu_switch_stop_ok(mock_run: MagicMock) -> None:
    from ashare_lab.research.batch_experiment import _gpu_switch
    mock_run.return_value = MagicMock(returncode=0)
    _gpu_switch("admin@192.168.100.11", "training")
    cmd = mock_run.call_args[0][0]
    assert "gpu-switch" in " ".join(cmd)
    assert "training" in " ".join(cmd)


@patch("subprocess.run")
def test_gpu_switch_start_ok(mock_run: MagicMock) -> None:
    from ashare_lab.research.batch_experiment import _gpu_switch
    mock_run.return_value = MagicMock(returncode=0)
    _gpu_switch("admin@192.168.100.11", "ollama")
    cmd = mock_run.call_args[0][0]
    assert "ollama" in " ".join(cmd)


@patch("subprocess.run")
def test_gpu_switch_failure_warns(mock_run: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    import logging
    from ashare_lab.research.batch_experiment import _gpu_switch
    mock_run.return_value = MagicMock(returncode=1, stderr="access denied")
    with caplog.at_level(logging.WARNING):
        _gpu_switch("admin@192.168.100.11", "training")
    assert "failed" in caplog.text


@patch("ashare_lab.research.batch_experiment._gpu_switch")
@patch("ashare_lab.research.batch_experiment._run_matrix_inner")
@patch("ashare_lab.research.batch_experiment.ensure_gpu_online")
def test_run_matrix_stops_ollama_and_restarts(
    mock_gpu: MagicMock,
    mock_inner: MagicMock,
    mock_ctl: MagicMock,
    tmp_path: Path,
) -> None:
    """run_matrix stops ollama before training, restarts after."""
    from ashare_lab.research.batch_experiment import run_matrix
    run_matrix(str(tmp_path), str(tmp_path / "out"))
    actions = [c[0][1] for c in mock_ctl.call_args_list]
    assert actions == ["training", "ollama"]


@patch("ashare_lab.research.batch_experiment._gpu_switch")
@patch("ashare_lab.research.batch_experiment._run_matrix_inner", side_effect=RuntimeError("boom"))
@patch("ashare_lab.research.batch_experiment.ensure_gpu_online")
def test_run_matrix_restarts_ollama_on_crash(
    mock_gpu: MagicMock,
    mock_inner: MagicMock,
    mock_ctl: MagicMock,
    tmp_path: Path,
) -> None:
    """Ollama service restarts even if training crashes (finally block)."""
    from ashare_lab.research.batch_experiment import run_matrix
    with pytest.raises(RuntimeError, match="boom"):
        run_matrix(str(tmp_path), str(tmp_path / "out"))
    start_calls = [c for c in mock_ctl.call_args_list if c[0][1] == "ollama"]
    assert len(start_calls) == 1
