"""Date overrides are research artifacts, not scheduled production checkpoints."""
from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import logging
from pathlib import Path
import sys

import pytest

from ashare_lab.research import smoke_test, train, windows

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ashare_lab/research/train.py"
CONFIG = {
    "walk_forward": {"step_months": 6, "train_start": "2018-01-01", "train_window_years": 3},
    "model": {"type": "tra", "handler": "alpha158", "routing": {"num_states": 3}},
}
BASE = {
    "window_id": 11, "train_start": "2022-12-31", "train_end": "2025-12-31",
    "valid_start": "2026-01-01", "valid_end": "2026-06-30",
    "test_start": "2026-07-01", "test_end": "2026-11-06", "is_complete": False,
}


def _cli(argv, monkeypatch, repo_root=ROOT, config_value=None, on_fit=None):
    tree = ast.parse(SOURCE.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.If) and
                ast.unparse(node.test) == "__name__ == '__main__'")
    calls = []
    def fit_boundary(window, **kw):
        calls.append((copy.deepcopy(window), kw))
        if on_fit is not None:
            on_fit(window, **kw)

    ns = {
        "__name__": "__main__", "__file__": str(repo_root / "ashare_lab/research/train.py"), "Path": Path,
        "log": logging.getLogger("cli-test"), "train_window": fit_boundary,
        "CHECKPOINT_SUFFIXES": train.CHECKPOINT_SUFFIXES,
    }
    frozen_config = copy.deepcopy(CONFIG if config_value is None else config_value)
    from ashare_lab import config
    monkeypatch.setattr(config, "load_config", lambda: copy.deepcopy(frozen_config))
    monkeypatch.setattr(windows, "load_config", lambda: copy.deepcopy(frozen_config))
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    monkeypatch.setattr(smoke_test, "get_all_windows", lambda: [dict(BASE), dict(BASE, window_id=10)])
    monkeypatch.setattr(smoke_test, "get_live_window", lambda: dict(BASE, window_id="live"))
    monkeypatch.setattr(smoke_test, "get_window", lambda step: dict(BASE))
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2026, 11, 6))
    # Execute the real CLI body; only the expensive fit boundary is replaced.
    exec(compile(ast.Module(body=main.body, type_ignores=[]), str(SOURCE), "exec"), ns)
    return calls


@pytest.mark.parametrize("mode", [["--live"], ["--serve-now", "10"]])
def test_rebase_refuses_production_modes(tmp_path, monkeypatch, mode, capsys):
    with pytest.raises(SystemExit) as error:
        _cli([*mode, "--train-end", "2026-05-31", "--valid-months", "1",
              "--exp-dir", str(tmp_path)], monkeypatch)
    assert error.value.code == 2
    assert "--live and --serve-now cannot be rebased" in capsys.readouterr().err
    assert not (tmp_path / "models").exists()


def test_rebase_requires_explicit_experiment_output(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as error:
        _cli(["--windows", "11", "--train-end", "2026-05-31", "--valid-months", "1"],
             monkeypatch, repo_root=tmp_path)
    assert error.value.code == 2


@pytest.mark.parametrize("selection", [[], ["--windows", "10,11"], ["--windows", ""], ["--windows", " "]])
def test_rebase_requires_one_explicit_window(tmp_path, monkeypatch, selection):
    with pytest.raises(SystemExit) as error:
        _cli([*selection, "--train-end", "2026-05-31", "--valid-months", "1",
              "--exp-dir", str(tmp_path)], monkeypatch)
    assert error.value.code == 2


def test_rebase_artifact_and_manifest_keep_full_fit_contract(tmp_path, monkeypatch):
    calls = _cli([
        "--windows", "11", "--train-end", "2026-05-31", "--valid-months", "1",
        "--test-months", "5", "--exp-dir", str(tmp_path),
    ], monkeypatch)
    assert len(calls) == 1
    window, kw = calls[0]
    assert str(window["window_id"]).startswith("experiment-"), "scheduled identity must not survive rebasing"
    assert window["train_start"] == "2023-05-31"
    assert window["train_end"] == "2026-05-31"
    assert window["test_start"] == "2026-07-01"
    assert kw["exp_dir"] != ROOT
    assert kw["exp_dir"].is_relative_to(tmp_path)
    manifest = kw["exp_dir"] / "models" / f"w{window['window_id']}.window.json"
    data = json.loads(manifest.read_text())
    assert data["window"] == window
    assert data["source_window_id"] == 11
    assert data["universe"] == "csi1000"
    assert data["config"] == CONFIG
    assert kw["universe"] == data["universe"]


def test_plain_scheduled_selection_preserves_id(tmp_path, monkeypatch):
    calls = _cli(["--windows", "11", "--exp-dir", str(tmp_path)], monkeypatch)
    assert calls[0][0] == BASE
    assert calls[0][1]["exp_dir"] == tmp_path


def test_explicit_repo_output_is_refused_for_rebasing(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as error:
        _cli(["--windows", "11", "--valid-months", "1", "--exp-dir", str(tmp_path)],
             monkeypatch, repo_root=tmp_path)
    assert error.value.code == 2


def test_different_fit_contracts_get_different_artifacts(tmp_path, monkeypatch):
    first = _cli(["--windows", "11", "--train-end", "2026-04-30", "--valid-months", "1",
                  "--exp-dir", str(tmp_path)], monkeypatch)
    second = _cli(["--windows", "11", "--train-end", "2026-05-31", "--valid-months", "1",
                   "--exp-dir", str(tmp_path)], monkeypatch)
    assert first[0][0]["window_id"] != second[0][0]["window_id"]
    assert first[0][1]["exp_dir"] != second[0][1]["exp_dir"]
    assert first[0][1]["exp_dir"].name == "experiment-" + first[0][0]["window_id"].split("-", 1)[1]
    assert second[0][1]["exp_dir"].name == "experiment-" + second[0][0]["window_id"].split("-", 1)[1]


def test_scheduled_predictor_refuses_research_checkpoint(tmp_path):
    from ashare_lab.research.predict import _resolve_model_path

    artifact = tmp_path / "wexperiment-isolated.pt"
    with pytest.raises(ValueError, match="research-only checkpoint"):
        _resolve_model_path("2026-08-03", artifact, [BASE], {}, tmp_path)


@pytest.mark.parametrize("layout", [
    "explicit-alias", "scheduled", "fallback", "live", "copy-meta", "copy-original-meta",
])
def test_final_resolved_research_identity_is_refused(tmp_path, monkeypatch, layout):
    from ashare_lab.research import predict
    from ashare_lab.research.model_meta import write_model_meta

    models = tmp_path / "models"
    models.mkdir()
    research = tmp_path / "wexperiment-recorded.pt"
    research.write_bytes(b"isolated non-loadable checkpoint marker")
    explicit = None
    trade_date = "2026-08-03"
    if layout == "explicit-alias":
        explicit = models / "alias.pt"
        explicit.symlink_to(research)
    elif layout in ("copy-meta", "copy-original-meta"):
        explicit = models / "latest.pt"
        explicit.write_bytes(research.read_bytes())
        write_model_meta(explicit, "2026-05-31", "wexperiment-recorded")
        if layout == "copy-original-meta":
            metadata = models / "latest.meta.json"
            row = json.loads(metadata.read_text())
            row["model_file"] = research.name
            metadata.write_text(json.dumps(row))
    elif layout == "scheduled":
        (models / "w11.pt").symlink_to(research)
    elif layout == "fallback":
        (models / "latest.pt").symlink_to(research)
    else:
        (models / "wlive.pt").symlink_to(research)
        trade_date = "2026-11-06"
    monkeypatch.setattr(predict, "_send_alert", lambda text: None)
    monkeypatch.setattr(smoke_test, "latest_trading_day", lambda: dt.date(2026, 11, 6))
    monkeypatch.setattr(smoke_test, "get_live_window", lambda: dict(BASE, window_id="live"))
    with pytest.raises(ValueError, match="research-only checkpoint"):
        predict._resolve_model_path(trade_date, explicit, [dict(BASE)], {}, models)


@pytest.mark.parametrize("identity_field,other_ordinary", [
    ("window_id", False), ("window_id", True), ("model_file", False), ("model_file", True),
])
@pytest.mark.parametrize("route", ["explicit", "scheduled", "fallback", "after-last", "live"])
@pytest.mark.parametrize("record", [
    "alias-stem", "alias-shared", "target-stem", "target-shared", "alias-manifest", "target-manifest", "ordinary",
])
def test_alias_and_target_provenance_is_seen_on_every_route(
    tmp_path, monkeypatch, route, record, identity_field, other_ordinary,
):
    from ashare_lab.research import predict

    models = tmp_path / "serving"
    storage = tmp_path / "storage"
    models.mkdir()
    storage.mkdir()
    target = storage / "opaque.pt"
    target.write_bytes(b"isolated marker, never load")
    alias_name = {"explicit": "alias.pt", "scheduled": "w11.pt", "live": "wlive.pt"}.get(route, "latest.pt")
    alias = models / alias_name
    alias.symlink_to(target)
    ordinary = {"window_id": "w11", "model_file": "opaque.pt", "train_date": "2020-01-01"}
    # Ordinary preferred records must not hide another record's identity.
    alias.with_suffix(".meta.json").write_text(json.dumps(ordinary))
    target.with_suffix(".meta.json").write_text(json.dumps(ordinary))
    descriptor = dict(ordinary) if other_ordinary else {}
    descriptor[identity_field] = "wexperiment-known.pt" if identity_field == "model_file" else "wexperiment-known"
    if other_ordinary:
        descriptor["train_date"] = "invalid"
    locations = {
        "alias-stem": alias.with_suffix(".meta.json"),
        "alias-shared": models / "meta.json",
        "target-stem": target.with_suffix(".meta.json"),
        "target-shared": storage / "meta.json",
        "alias-manifest": alias.with_suffix(".window.json"),
        "target-manifest": target.with_suffix(".window.json"),
    }
    if record.endswith("manifest"):
        locations[record].write_bytes(b"unreadable manifest, identity by existence only\xff")
    elif record != "ordinary":
        locations[record].write_text(json.dumps(descriptor))
    monkeypatch.setattr(predict, "_send_alert", lambda text: None)
    monkeypatch.setattr(smoke_test, "latest_trading_day", lambda: dt.date(2026, 11, 6))
    monkeypatch.setattr(smoke_test, "get_live_window", lambda: dict(BASE, window_id="live"))
    trade_date = {"after-last": "2026-11-07", "live": "2026-11-06"}.get(route, "2026-08-03")
    explicit = alias if route == "explicit" else None
    if record == "ordinary":
        selected, _ = predict._resolve_model_path(trade_date, explicit, [dict(BASE)], {}, models)
        assert selected.resolve() == target
    else:
        with pytest.raises(ValueError, match="research-only checkpoint"):
            predict._resolve_model_path(trade_date, explicit, [dict(BASE)], {}, models)
    assert target.read_bytes() == b"isolated marker, never load"


@pytest.mark.parametrize("identity_field", ["window_id", "model_file"])
@pytest.mark.parametrize("record", ["stem", "shared"])
@pytest.mark.parametrize("route", ["explicit", "scheduled", "fallback", "after-last", "live"])
def test_single_identity_field_survives_regular_checkpoint_copy(tmp_path, monkeypatch, route, record, identity_field):
    from ashare_lab.research import predict

    models = tmp_path / "serving"
    models.mkdir()
    original = tmp_path / "wexperiment-original.pt"
    original.write_bytes(b"isolated original, never load")
    name = {"explicit": "alias.pt", "scheduled": "w11.pt", "live": "wlive.pt"}.get(route, "latest.pt")
    selected = models / name
    selected.write_bytes(original.read_bytes())
    assert not selected.is_symlink()
    descriptor = {identity_field: original.name if identity_field == "model_file" else "wexperiment-original",
                  "train_date": "invalid"}
    location = selected.with_suffix(".meta.json") if record == "stem" else models / "meta.json"
    location.write_text(json.dumps(descriptor))
    before = {path.name: path.read_bytes() for path in models.iterdir()}
    monkeypatch.setattr(predict, "_send_alert", lambda text: None)
    monkeypatch.setattr(smoke_test, "latest_trading_day", lambda: dt.date(2026, 11, 6))
    monkeypatch.setattr(smoke_test, "get_live_window", lambda: dict(BASE, window_id="live"))
    trade_date = {"after-last": "2026-11-07", "live": "2026-11-06"}.get(route, "2026-08-03")
    with pytest.raises(ValueError, match="research-only checkpoint"):
        predict._resolve_model_path(trade_date, selected if route == "explicit" else None, [dict(BASE)], {}, models)
    assert {path.name: path.read_bytes() for path in models.iterdir()} == before
    assert original.read_bytes() == b"isolated original, never load"


@pytest.mark.parametrize("changed", ["routing", "handler", "universe"])
def test_config_and_universe_change_experiment_identity(tmp_path, monkeypatch, changed):
    args = ["--windows", "11", "--train-end", "2026-05-31", "--valid-months", "1",
            "--exp-dir", str(tmp_path)]
    config = copy.deepcopy(CONFIG)
    first = _cli(args, monkeypatch)
    if changed == "routing":
        config["model"]["routing"]["num_states"] = 7
    elif changed == "handler":
        config["model"]["handler"] = "alpha360"
    else:
        args += ["--universe", "csi500"]
    second = _cli(args, monkeypatch, config_value=config)
    assert first[0][0]["window_id"] != second[0][0]["window_id"]
    assert first[0][1]["exp_dir"] != second[0][1]["exp_dir"]
    window, kw = second[0]
    manifest = kw["exp_dir"] / "models" / f"w{window['window_id']}.window.json"
    data = json.loads(manifest.read_text())
    assert data["config"] == config
    assert data["universe"] == kw["universe"]


@pytest.mark.parametrize("model_type, suffix", [
    ("tra", ".pt"), ("alstm", ".pt"), ("lgbm", ".pkl"), ("densemble", ".pkl"),
])
def test_existing_checkpoint_reuse_does_not_rewrite_provenance(tmp_path, monkeypatch, model_type, suffix):
    config = copy.deepcopy(CONFIG)
    config["model"]["type"] = model_type
    args = ["--windows", "11", "--train-end", "2026-05-31", "--valid-months", "1",
            "--exp-dir", str(tmp_path)]
    first = _cli(args, monkeypatch, config_value=config)
    window, kw = first[0]
    parent = kw["exp_dir"] / "models"
    artifact = parent / f"w{window['window_id']}{suffix}"
    artifact.write_bytes(b"model fixture, never load")
    manifest = parent / f"w{window['window_id']}.window.json"
    before = manifest.read_bytes()
    import os
    os.utime(manifest, ns=(1_000_000_000, 1_000_000_000))
    stamp = manifest.stat().st_mtime_ns
    second = _cli(args, monkeypatch, config_value=config)
    assert second == []
    assert manifest.read_bytes() == before
    assert manifest.stat().st_mtime_ns == stamp
    # Refuse to assign a new provenance to bytes whose history is not known.
    manifest.write_text('{"sentinel": "preserve existing provenance"}')
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="contract"):
        _cli(args, monkeypatch, config_value=config)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("alias_kind", ["directory", "manifest", "checkpoint", "hardlink", "metadata"])
def test_experiment_subtree_aliases_are_refused_before_writes(tmp_path, monkeypatch, alias_kind):
    args = ["--windows", "11", "--train-end", "2026-05-31", "--valid-months", "1",
            "--exp-dir", str(tmp_path)]
    first = _cli(args, monkeypatch)
    window, kw = first[0]
    parent = kw["exp_dir"] / "models"
    manifest = parent / f"w{window['window_id']}.window.json"
    shared = tmp_path / "unrelated"
    shared.mkdir()
    sentinel = shared / "record"
    sentinel.write_bytes(b"not this experiment")
    if alias_kind == "directory":
        manifest.unlink()
        parent.rmdir()
        parent.symlink_to(shared, target_is_directory=True)
    elif alias_kind == "manifest":
        manifest.unlink()
        manifest.symlink_to(sentinel)
    elif alias_kind == "checkpoint":
        (parent / f"w{window['window_id']}.pt").symlink_to(sentinel)
    elif alias_kind == "hardlink":
        manifest.unlink()
        manifest.hardlink_to(sentinel)
    else:
        (parent / "meta.json").symlink_to(sentinel)
    with pytest.raises(ValueError, match="alias"):
        _cli(args, monkeypatch)
    assert sentinel.read_bytes() == b"not this experiment"
    assert sorted(p.name for p in shared.iterdir()) == ["record"]
