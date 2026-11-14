"""Tests for sync_gpu_data in scripts/ashare-data-update.sh."""

import re
import subprocess
import textwrap
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "ashare-data-update.sh")

# Preamble: source the script with testing guard, override externals.
# wake_gpu is stubbed by default: the real one sends a magic packet and
# polls for 200s.  Tests that care about the wake override the stub.
_PREAMBLE = textwrap.dedent(f"""\
    export ASHARE_DATA_UPDATE_TESTING=1
    export HOME="${{TMPDIR:-/tmp}}"
    source {SCRIPT}
    wake_gpu() {{ return 0; }}
""")


def _run(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", _PREAMBLE + snippet],
        capture_output=True, text=True, timeout=10,
    )


class TestSyncFunctionExists:
    def test_declared(self):
        r = _run("declare -f sync_gpu_data >/dev/null")
        assert r.returncode == 0, r.stderr


class TestSyncCallOrder:
    def test_tar_scp_ssh_in_order(self, tmp_path):
        log = tmp_path / "call.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            tar() {{ echo "tar" >> {log}; command tar "$@"; }}
            scp() {{ echo "scp" >> {log}; return 0; }}
            ssh() {{ echo "ssh" >> {log}; return 0; }}
            export -f tar scp ssh
            sync_gpu_data
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        calls = log.read_text().strip().splitlines()
        assert calls == ["tar", "scp", "ssh"]


class TestSyncHappyPath:
    def test_creates_and_cleans_tarball(self, tmp_path):
        """Verify tar creates the archive and trap cleans it on return."""
        r = _run(textwrap.dedent("""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            scp() { return 0; }
            ssh() { return 0; }
            export -f scp ssh
            sync_gpu_data
            [ ! -f "$HOME/.cache/ashare-sync/cn_data_sync.tar.gz" ] && [ ! -f /tmp/cn_data_sync.tar.gz ] && echo CLEANED
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        assert "CLEANED" in r.stdout


class TestSyncScratchLocation:
    def test_tar_uses_disk_cache_not_tmp(self, tmp_path):
        """Verify tar command targets ~/.cache/ashare-sync and not /tmp."""
        log = tmp_path / "tar_target.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            tar() {{
                echo "$2" > {log}
                command tar "$@"
            }}
            scp() {{ return 0; }}
            ssh() {{ return 0; }}
            export -f tar scp ssh
            sync_gpu_data
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        target = log.read_text().strip()
        assert target.endswith(".cache/ashare-sync/cn_data_sync.tar.gz")
        assert not target.startswith("/tmp/cn_data_sync.tar.gz")


class TestSyncTarFailure:
    def test_returns_nonzero(self):
        r = _run(textwrap.dedent("""\
            tar() { return 1; }
            export -f tar
            sync_gpu_data
        """))
        assert r.returncode != 0


class TestSyncScpFailure:
    def test_returns_nonzero(self, tmp_path):
        r = _run(textwrap.dedent("""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            scp() { return 1; }
            ssh() { echo "BUG: ssh should not be called"; return 0; }
            export -f scp ssh
            sync_gpu_data
        """))
        assert r.returncode != 0
        assert "ssh should not be called" not in r.stdout


class TestSyncSshFailure:
    def test_returns_nonzero(self, tmp_path):
        r = _run(textwrap.dedent("""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            scp() { return 0; }
            ssh() { return 1; }
            export -f scp ssh
            sync_gpu_data
        """))
        assert r.returncode != 0


class TestSyncPlacement:
    def test_function_outside_guard(self):
        """sync_gpu_data must be defined before the testing guard."""
        src = Path(SCRIPT).read_text()
        func_pos = src.index("sync_gpu_data()")
        guard_pos = src.index('ASHARE_DATA_UPDATE_TESTING')
        assert func_pos < guard_pos, (
            "sync_gpu_data must be defined outside the testing guard"
        )


class TestSyncWakesGpu:
    """The GPU sleeps between runs; without a wake, scp hits 'No route to
    host', the sync is skipped, and the 18:00 pipeline records a stale run
    (2026-09-02 incident).  These tests pin the wake to the sync path."""

    def test_wake_called_before_scp(self, tmp_path):
        log = tmp_path / "order.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            wake_gpu() {{ echo "wake" >> {log}; return 0; }}
            scp() {{ echo "scp" >> {log}; return 0; }}
            ssh() {{ echo "ssh" >> {log}; return 0; }}
            export -f wake_gpu scp ssh
            sync_gpu_data
        """))
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
        calls = log.read_text().strip().splitlines()
        assert calls[0] == "wake", f"wake must run first, got {calls}"
        assert "scp" in calls and calls.index("wake") < calls.index("scp")

    def test_wake_failure_aborts_sync(self, tmp_path):
        """A sleeping GPU must fail the sync loudly, not scp into the void."""
        log = tmp_path / "calls.log"
        r = _run(textwrap.dedent(f"""\
            QLIB_DIR=$(mktemp -d)
            mkdir -p "$QLIB_DIR/cn_data"
            touch "$QLIB_DIR/cn_data/dummy.bin"
            wake_gpu() {{ return 1; }}
            scp() {{ echo "scp" >> {log}; return 0; }}
            ssh() {{ echo "ssh" >> {log}; return 0; }}
            export -f wake_gpu scp ssh
            sync_gpu_data
        """))
        assert r.returncode != 0, "sync must fail when the GPU will not wake"
        assert not log.exists(), "no transfer may be attempted after wake failure"


class TestSharedWakeLib:
    """One wake implementation, sourced by both scripts (no copy-paste)."""

    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"
    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"

    def test_lib_defines_wake_gpu(self):
        assert "wake_gpu()" in self.LIB.read_text()

    def test_both_scripts_source_the_lib(self):
        for script in (Path(SCRIPT), self.PIPELINE):
            assert "lib/gpu-wake.sh" in script.read_text(), (
                f"{script.name} must source the shared wake lib"
            )

    def test_pipeline_has_no_inline_wake(self):
        """The pipeline's own WoL block was removed, not left duplicated."""
        src = self.PIPELINE.read_text()
        assert 'wol "$GPU_MAC"' not in src, (
            "inline wol call still present -- wake logic is duplicated"
        )


class TestGpuCodeSyncShipsModelMeta:
    """predict.py imports model_meta; GPU sync must ship the module.

    gpu-win latest.pt is a regular-file copy. Without model_meta.py on
    that host the staleness gate always falls back to mtime.
    """

    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"
    DEPLOY = Path(SCRIPT).resolve().parent / "deploy.sh"

    def test_pipeline_sync_copies_model_meta(self):
        src = self.PIPELINE.read_text()
        fn = src[src.index("sync_code_to_gpu()"): src.index("try_gpu_inference()")]
        assert "model_meta.py" in fn, (
            "sync_code_to_gpu must copy model_meta.py with predict.py"
        )
        assert "predict.py" in fn

    def test_deploy_sync_copies_model_meta(self):
        src = self.DEPLOY.read_text()
        staging = src[src.index("Staging files for GPU deploy"): src.index("Section 8")]
        assert "model_meta.py" in staging
        assert "predict.py" in staging


class TestGpuLastConsumer:
    """After a DEMAND_START reboot, llama is not running. Restore must
    follow the last durable consumer, not the ollama default."""

    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"
    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"
    SWITCH = Path(SCRIPT).resolve().parent / "gpu-switch.bat"

    def _run_lib(self, snippet: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", f"source {self.LIB}\n" + snippet],
            capture_output=True, text=True, timeout=10,
        )

    def test_live_llama_wins_over_stale_file(self):
        r = self._run_lib(
            'resolve_gpu_restore_target "llama-server.exe 1234" "ollama"; echo RC=$?'
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-2:] == ["llama-server", "RC=0"] or (
            r.stdout.strip().splitlines()[-1] == "llama-server"
        )

    def test_reboot_uses_last_llama_when_process_gone(self):
        r = self._run_lib('resolve_gpu_restore_target "" "llama-server"')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == "llama-server"

    def test_reboot_uses_last_ollama(self):
        r = self._run_lib('resolve_gpu_restore_target "" "ollama\\r"')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == "ollama"

    def test_unknown_or_empty_last_stays_ollama(self):
        r = self._run_lib('resolve_gpu_restore_target "" "training"')
        assert r.stdout.strip().splitlines()[-1] == "ollama"
        r2 = self._run_lib('resolve_gpu_restore_target "" ""')
        assert r2.stdout.strip().splitlines()[-1] == "ollama"

    def test_pipeline_calls_resolver_not_hardcoded_default(self):
        src = self.PIPELINE.read_text()
        fn = src[src.index("try_gpu_inference()"): src.index("gpu-switch.bat training")]
        assert "resolve_gpu_restore_target" in fn
        assert "gpu_restore_target=\"ollama\"" not in fn

    def test_restore_target_variable_is_actually_used(self):
        """Setting the variable is worthless if the restore call ignores it.

        An earlier version of this guard only checked the assignment, which
        would have stayed green if someone hardcoded the restore argument.
        """
        src = self.PIPELINE.read_text()
        fn = self._function_body(src, "try_gpu_inference()")

        lines = fn.splitlines()
        switch_at = [
            i for i, ln in enumerate(lines)
            if "gpu-switch.bat" in ln and "ssh" in ln
        ]
        assert switch_at, "no gpu-switch invocation found"

        # Identify the restore call by position, not by shape. Checking
        # "any call uses the variable" is satisfied by the pre-inference
        # clear, so hardcoding the restore argument would slip through while
        # gpu_restore_target -- and with it the resolver and the dedicated
        # switch -- became dead weight.
        predict_at = min(
            i for i, ln in enumerate(lines)
            if "predict.py" in ln or "research.predict" in ln
        )
        after = [i for i in switch_at if i > predict_at]
        assert after, (
            "no gpu-switch call after predict; nothing restores the card"
        )

        restore_line = lines[min(after)]
        assert "${gpu_restore_target}" in restore_line, (
            f"the restore call after predict ignores gpu_restore_target, so "
            f"the dedicated switch has no effect: {restore_line.strip()}"
        )

    def test_preinference_clear_stops_llama_server_too(self):
        """Clearing the card before predict has to stop llama-server.

        Plain "training" stops only OllamaService, so llama-server kept all
        8 GiB and predict fell back to the second GPU index instead of
        running on a free card.
        """
        src = self.PIPELINE.read_text()
        fn = self._function_body(src, "try_gpu_inference()")

        literal = [
            ln for ln in fn.splitlines()
            if "gpu-switch.bat" in ln and "${gpu_restore_target}" not in ln
            and "ssh" in ln
        ]
        assert literal, "no literal pre-inference clear found"
        for ln in literal:
            assert "training-3080" in ln, (
                f"pre-inference clear uses a target that leaves llama-server "
                f"running: {ln.strip()}"
            )

    @staticmethod
    def _function_body(src, name):
        """Return a function body bounded by balanced braces.

        Slicing to the first line starting with '}' breaks the moment a
        top-level brace appears inside the body: the window shrinks and
        every check inside it silently passes on an empty string.
        """
        start = src.index(name)
        open_at = src.index("{", start)
        depth = 0
        for i in range(open_at, len(src)):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    body = src[open_at + 1:i]
                    assert body.strip(), f"{name} body came back empty"
                    return body
        raise AssertionError(f"unbalanced braces in {name}")

    def test_switch_off_keeps_the_resolver_result(self):
        """With the switch off, the resolved consumer has to survive.

        The guard overwrites a value the resolver computed just above it, so
        an unconditional assignment would silently discard that work and
        make ASHARE_GPU_DEDICATED=0 impossible to honour.
        """
        src = self.PIPELINE.read_text()
        fn = self._function_body(src, "try_gpu_inference()")

        assert "resolve_gpu_restore_target" in fn, (
            "resolver call is gone from try_gpu_inference; the dedicated "
            "switch has nothing to fall back to"
        )
        assert "ASHARE_GPU_DEDICATED" in fn, "dedicated switch is gone"
        resolve_at = fn.index("resolve_gpu_restore_target")
        guard_at = fn.index("ASHARE_GPU_DEDICATED")
        assert resolve_at < guard_at, (
            "guard runs before the resolver, so the resolved value is lost"
        )

        # Start the window at the line opening the guard, not at the variable
        # name: the "if" keyword sits to the left of it.
        line_start = fn.rfind("\n", 0, guard_at) + 1
        guard = fn[line_start:]
        override = re.search(r"gpu_restore_target=(\S+)", guard)
        assert override, "dedicated branch assigns nothing"

        # The override must sit inside a conditional, not run every time.
        head = guard[:override.start()]
        assert "if " in head and "then" in head, (
            "override is unconditional; ASHARE_GPU_DEDICATED=0 is ignored"
        )
        closing = guard[override.end():]
        assert re.search(r"^\s*fi\s*$", closing, re.MULTILINE), (
            "no 'fi' after the override; the branch never closes"
        )

    def test_dedicated_mode_releases_the_card_not_just_ollama(self):
        """The card is dedicated to this pipeline, so after inference it has
        to come back free.

        This host has one GPU and llama-server is what holds its memory.
        gpu-switch.bat has two stop targets and only one of them touches
        llama-server: plain "training" stops OllamaService alone and says so
        in its own echo, leaving all 8 GiB pinned. Accepting both names here
        would hide exactly that bug, so only the target that stops
        llama-server counts.

        The window is bounded by the function's own closing brace and starts
        at the guard, so moving either end cannot silently empty it.
        """
        src = self.PIPELINE.read_text()
        fn = self._function_body(src, "try_gpu_inference()")
        assert "ASHARE_GPU_DEDICATED" in fn, "dedicated switch missing"

        guard = fn[fn.index("ASHARE_GPU_DEDICATED"):]
        assigned = re.findall(r"gpu_restore_target=(\S+)", guard)
        assert assigned, "dedicated branch assigns no restore target"

        for value in assigned:
            bare = value.strip('"').strip("'")
            assert bare == "training-3080", (
                f"dedicated mode restores to {value!r}; only 'training-3080' "
                f"stops llama-server and actually frees the card"
            )

    def test_switch_writes_durable_consumer_for_serve_targets(self):
        src = self.SWITCH.read_text()
        assert "gpu-last-consumer.txt" in src
        ollama = src[src.index("\n:ollama\n"): src.index("\n:llama-server\n")]
        llama = src[src.index("\n:llama-server\n"): src.index("\n:training\n")]
        training = src[src.index("\n:training\n"): src.index("\n:training-3080\n")]
        train3080 = src[src.index("\n:training-3080\n"):]
        assert "gpu-last-consumer.txt" in ollama
        assert "gpu-last-consumer.txt" in llama
        assert "gpu-last-consumer.txt" not in training
        assert "gpu-last-consumer.txt" not in train3080


class TestRetrainSkipsServeNowLive:
    """The cadence retrain trains the freshest window, never --force over the
    whole walk-forward set.

    The rule used to depend on the live model's number (>= 100 meant it sat
    outside the walk-forward range).  Once the live model became w12 -- inside
    the range but still newest -- that gate would have switched every retrain
    to --force, which walks eleven windows for a refresh that needs one.  The
    step is now always derived, and the deploy gate compares train_date, so the
    model's name no longer decides anything.
    """

    RETRAIN = Path(SCRIPT).resolve().parent / "ashare-retrain.sh"

    def test_serve_now_step_is_derived_not_hardcoded(self):
        """get_all_windows() stops at the last window whose test period has
        started, so the freshest trainable step is always its length. A
        literal step number silently trains a stale window once the data
        grows past the next window boundary."""
        src = self.RETRAIN.read_text()
        assert "SERVE_NOW_STEP=11" not in src, (
            "step is hardcoded; derive it from len(get_all_windows())"
        )
        assert "get_all_windows" in src

    def test_serve_now_guard_before_train(self):
        src = self.RETRAIN.read_text()
        train_at = src.index("py -m ashare_lab.research.train $TRAIN_ARGS")
        guard_at = src.index("cannot derive the newest window step")
        assert guard_at < train_at, "the abort must precede the training call"
        # The deploy gate still branches on -ge 100 to recognise an out-of-band
        # serve-now model whose number cannot be compared numerically; that is
        # a different question. Only the block that chooses the training shape
        # must be free of the model-name test.
        shape = src[src.index("# Always retrain the freshest window"):train_at]
        assert "-ge 100" not in shape, (
            "the training shape still depends on the model name; w12 would "
            "have flipped every retrain to --force over the whole history"
        )
        assert "--serve-now $SERVE_NOW_STEP --force" in shape
        fn_at = src.index("read_expected_live_model()")
        fn_end = src.index("\n}", fn_at) + 2
        fn = src[fn_at:fn_end]
        assert "awk" in fn
        assert "python3 -c" not in fn
        assert "2>/dev/null" not in fn

    def test_deploy_gate_judges_named_models_not_the_shared_meta(self):
        """The gate compares train dates of walk-forward checkpoints by name.

        It used to read models/meta.json. A live artifact writes there too, so
        whichever model wrote last would supply the dates for a comparison it
        does not belong to: the live model's train_date is yesterday, which
        makes every following walk-forward model look staler and shelves it.
        """
        src = self.RETRAIN.read_text()
        gate = src[src.index("# --- Step 3b"):src.index('if [ "$DEPLOY_SKIP" -eq 1 ]; then')]

        assert "open('meta.json')" not in gate, "gate still reads the shared meta"
        assert "w*.pt" in gate, "the producing model must be found by name"
        assert "${NEW_MODEL%.pt}.meta.json" in gate, "per-checkpoint meta not read"
        assert "NEW_TRAIN_DATE" in gate and "LIVE_TRAIN_DATE" in gate
        assert "cannot read both train dates" in gate, (
            "an unreadable pair must shelve, not fall through"
        )

    def test_alerts_do_not_interpolate_values_into_python(self):
        """Alert values read off gpu-win are data, not code. Embedding
        them in the inline python source lets a quote in meta.json break
        the alert, so the shelved-model warning is lost. Shell log lines
        are unaffected; only the python -c blocks are scanned."""
        src = self.RETRAIN.read_text()
        blocks = re.findall(r"python3 -c (.)\n(.*?)\n\1", src, re.S)
        assert blocks, "no inline python blocks found"
        for _quote, body in blocks:
            assert "$" not in body, (
                f"shell value interpolated into inline python:\n{body}"
            )
        assert src.count("os.environ[") >= 4


class TestWakeLibBehavior:
    """Direct tests of the shared lib, independent of the sync path."""

    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"

    def _run_lib(self, snippet: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", f"source {self.LIB}\n" + snippet],
            capture_output=True, text=True, timeout=10,
        )

    def test_missing_wol_binary_fails(self):
        """A missing wol used to print to stderr and carry on: the ping poll
        still succeeded against an already-awake host, so a box that never
        got a magic packet looked healthy.  Found by running the real path
        on a machine without wol installed."""
        r = self._run_lib(textwrap.dedent("""\
            PATH=/nonexistent
            command() { return 1; }
            ping() { return 0; }
            ssh() { return 0; }
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 5
        """))
        assert r.returncode != 0, (
            "wake must fail when wol is absent, not proceed to the ping poll"
        )
        assert "wol not installed" in r.stdout

    def test_wake_succeeds_when_host_answers(self):
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 5
        """))
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"

    def test_wol_send_failure_is_reported(self):
        """An unchecked wol turns 'the packet never went out' into the
        generic unreachable error 200 seconds later."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 1; }
            ping() { echo "BUG: ping ran after wol failed"; return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 5
        """))
        assert r.returncode != 0
        assert "wol failed" in r.stdout, r.stdout
        assert "BUG:" not in r.stdout, "must not poll after a failed send"

    def test_ping_poll_respects_the_budget(self):
        """The poll used to run a fixed 40 iterations whatever the caller
        asked for, so the budget could not bound the stall.  Comparing two
        budgets is what makes this a budget test: a fixed-count poll gives
        the same iteration count for both, a bounded one does not.

        Iterations are counted rather than seconds so the assertion does
        not depend on wall-clock timing; sleep is stubbed to keep the test
        fast, which means SECONDS advances by real time, not by the stub."""
        def run_with_budget(budget: str) -> tuple[str, int]:
            r = self._run_lib(textwrap.dedent(f"""\
                wol() {{ return 0; }}
                ping() {{ return 1; }}
                ssh() {{ return 1; }}
                iters=0
                sleep() {{ iters=$((iters+1)); command sleep 0.05; }}
                wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser {budget}
                echo "RC=$? ITERS=$iters"
            """))
            # The trailing echo sets the script's own status, so read
            # wake_gpu's return code from the output, not from returncode.
            iters = int(r.stdout.split("ITERS=")[1].split()[0])
            return r.stdout, iters

        small_out, small_iters = run_with_budget("1")
        large_out, large_iters = run_with_budget("3")

        assert "RC=1" in small_out, small_out
        assert "RC=1" in large_out, large_out
        assert large_iters > small_iters, (
            f"budget=1 ran {small_iters} iterations and budget=3 ran "
            f"{large_iters}; the poll length does not follow the budget"
        )


    def test_non_numeric_budget_rejected(self):
        """A non-numeric budget makes the deadline comparison error out and
        evaluate false, so the ping poll never runs and the wake reports
        unreachable without having waited at all.  (An EMPTY budget is not
        this case: ${4:-300} substitutes the default, which is correct.)"""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { echo "BUG: polled with a bad budget"; return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser "abc"
            echo "RC=$?"
        """))
        assert "RC=1" in r.stdout, r.stdout
        assert "budget must be a positive integer" in r.stdout
        assert "BUG:" not in r.stdout

    def test_zero_budget_rejected(self):
        """Zero passes a bare ^[0-9]+$ but makes the deadline false on the
        first pass, so no poll runs and the caller is told the host is
        unreachable -- a misleading message for a bad argument."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { echo "BUG: polled with a zero budget"; return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser 0
            echo "RC=$?"
        """))
        assert "RC=1" in r.stdout, r.stdout
        assert "budget must be a positive integer" in r.stdout, r.stdout
        assert "not reachable" not in r.stdout, (
            "a bad budget must not be reported as an unreachable host"
        )
        assert "BUG:" not in r.stdout

    def test_empty_budget_uses_the_default(self):
        """Documents the boundary the guard must NOT reject."""
        r = self._run_lib(textwrap.dedent("""\
            wol() { return 0; }
            ping() { return 0; }
            ssh() { return 0; }
            export -f wol ping ssh
            wake_gpu 10.0.0.1 aa:bb:cc:dd:ee:ff someuser ""
            echo "RC=$?"
        """))
        assert "RC=0" in r.stdout, r.stdout


class TestGpuIdentityIsSharedOnce:
    """GPU_HOST/MAC/USER live in the lib so a hardware swap is one edit."""

    LIB = Path(SCRIPT).resolve().parent / "lib" / "gpu-wake.sh"
    PIPELINE = Path(SCRIPT).resolve().parent / "ashare-pipeline.sh"

    def test_lib_defines_the_identity(self):
        src = self.LIB.read_text()
        for var in ("GPU_HOST=", "GPU_MAC=", "GPU_USER="):
            assert var in src, f"{var} must live in the shared lib"

    def test_callers_do_not_redefine_it(self):
        for script in (Path(SCRIPT), self.PIPELINE):
            src = script.read_text()
            for var in ("GPU_HOST=", "GPU_MAC=", "GPU_USER="):
                assert var not in src, (
                    f"{script.name} redefines {var}; the hardware address "
                    "would then have to be changed in lockstep across files"
                )

    def test_identity_is_defined_before_first_use(self):
        """The lib is sourced partway down the pipeline; if any GPU_* use
        preceded the source line the variable would be empty under set -u."""
        src = self.PIPELINE.read_text().splitlines()
        source_line = next(
            i for i, ln in enumerate(src) if "lib/gpu-wake.sh" in ln and "source" in ln
        )
        uses = [
            i for i, ln in enumerate(src)
            if ("$GPU_HOST" in ln or "${GPU_HOST}" in ln
                or "$GPU_MAC" in ln or "${GPU_USER}" in ln)
        ]
        assert uses, "expected the pipeline to use the shared identity"
        assert min(uses) > source_line, (
            f"GPU_* used at line {min(uses) + 1} before the lib is sourced "
            f"at line {source_line + 1}"
        )


class TestSyncFailureAlerts:
    """Sync failure used to exit 0 silently; the only visible symptom was a
    stale pipeline run 15 minutes later, pointing at the wrong layer."""

    def test_alert_invoked_on_sync_failure(self):
        src = Path(SCRIPT).read_text()
        idx = src.index("WARNING: GPU data sync failed")
        tail = src[idx:idx + 400]
        assert "alert.py" in tail, (
            "GPU sync failure must raise an alert, not just echo a warning"
        )
        assert "gpu_data_sync" in tail, (
            "alert must name the failing stage so it is not confused "
            "with a data_update or pipeline failure"
        )

    def test_alert_gets_the_sync_log_not_the_fetch_log(self):
        """alert.py reports the tail of the log it is handed.  Passing
        STDERR_LOG (fetch-today's output) means the alert names a failed
        sync without saying which step broke."""
        src = Path(SCRIPT).read_text()
        idx = src.index("gpu_data_sync")
        call = src[idx:idx + 200]
        assert "SYNC_LOG" in call, "alert must carry the sync's own log"
        assert "STDERR_LOG" not in call, (
            "STDERR_LOG holds fetch-today output, not sync diagnostics"
        )

    def test_sync_failure_detected_without_relying_on_pipefail(self):
        """The sync output is piped through tee, and tee always exits 0.
        Reading the pipeline status would make the alert depend on pipefail
        staying set -- a future refactor into a subshell or `bash -c` would
        silently disarm it.  PIPESTATUS[0] is explicit about which command's
        status matters."""
        src = Path(SCRIPT).read_text()
        assert "PIPESTATUS[0]" in src, (
            "sync failure must be read from PIPESTATUS, not the pipeline "
            "status that tee overwrites"
        )
        assert "if ! sync_gpu_data 2>&1 | tee" not in src, (
            "pipeline-status form reintroduces the pipefail dependency"
        )

    def test_pipestatus_form_detects_failure(self):
        """Behavioural check that the chosen form works even with pipefail
        off -- the point of using PIPESTATUS in the first place."""
        r = subprocess.run(
            ["bash", "-c", textwrap.dedent("""\
                set +o pipefail
                f() { return 1; }
                f 2>&1 | tee /dev/null
                rc=${PIPESTATUS[0]}
                [ "$rc" -ne 0 ] && echo DETECTED || echo MASKED
            """)],
            capture_output=True, text=True, timeout=10,
        )
        assert "DETECTED" in r.stdout, r.stdout


class TestRetrainPointsTheGpuAtItsOwnModel:
    """The GPU host must load the model this run deployed.

    It runs the nightly prediction and, past the last window, resolves
    latest.pt. On Windows that file is a real copy rather than the symlink W1
    uses, so a retrain that only relinks on W1 leaves the inference host on the
    previous model -- which is what happened when w12 deployed and the GPU kept
    serving w115.
    """

    RETRAIN = Path(__file__).resolve().parents[1] / "scripts" / "ashare-retrain.sh"

    def test_the_gpu_copy_of_latest_is_updated(self):
        src = self.RETRAIN.read_text()
        assert "copy /Y $DEPLOYED_MODEL latest.pt" in src
        assert "GPU latest.pt ->" in src

    def test_the_relink_sits_in_the_deploying_branch(self):
        src = self.RETRAIN.read_text()
        skip_at = src.index('log "Shelved model: skipping Step 4')
        relink_at = src.index("copy /Y $DEPLOYED_MODEL latest.pt")
        restart_at = src.index("# --- Step 6: Restart llama-server")
        assert skip_at < relink_at < restart_at, (
            "a shelved model must not be pushed to the GPU, and the copy has to "
            "happen before the server restart"
        )


class TestRetrainTrainsTheLiveWindow:
    """A retrain must leave a model that has actually seen recent data.

    The walk-forward window cannot: its training span ends before its own
    validation and test spans. The live window trains three rolling years
    ending on the last trading day, and it is promoted only on the same run
    that passed the deploy gate.
    """

    RETRAIN = Path(__file__).resolve().parents[1] / "scripts" / "ashare-retrain.sh"

    def test_the_live_window_is_trained(self):
        src = self.RETRAIN.read_text()
        assert "py -m ashare_lab.research.train --live --force" in src
        assert "Training live window" in src

    def test_the_live_artifact_is_promoted_on_both_hosts(self):
        src = self.RETRAIN.read_text()
        assert "wlive.pt" in src
        assert 'ln -sf "wlive.pt"' in src
        assert "copy /Y wlive.pt latest.pt" in src

    def test_a_shelved_run_does_not_promote_a_live_model(self):
        """The live step sits in the else branch of the deploy gate.

        The else body is indented one level. A marker that has drifted to
        column zero still passes any ordering check while running on a shelved
        cycle -- promoting a model the gate just refused.
        """
        import re

        src = self.RETRAIN.read_text()
        skip_at = src.index('log "Shelved model: skipping Step 4')
        live_at = src.index("Training live window")
        restart_at = src.index("# --- Step 6: Restart llama-server")
        assert skip_at < live_at < restart_at
        assert re.search(r'^    log "Training live window\.\.\."$', src, re.M), (
            "the live step must stay inside the deploying branch"
        )
        gate = src.index('if [ "$DEPLOY_SKIP" -eq 1 ]; then')
        gate_else = src.index("else", gate)
        assert gate_else < live_at < restart_at


class TestRetrainRunsOnACalendar:
    """The retrain is scheduled, not triggered by a single period's IC.

    Over 30 sessions this book's daily IC has sd 0.23 with lag-1 correlation
    0.74, so "three consecutive days below 0.02" measured the sign of a noisy
    series. Fetching the value also removed the only consumer of
    IC_HISTORY, which must not be resurrected as a gate by a later edit.
    """

    def test_the_script_has_no_ic_gate(self):
        script = Path("scripts/ashare-retrain.sh").read_text()

        assert "IC gate" not in script
        assert "IC_THRESHOLD" not in script
        assert "CONSECUTIVE_DAYS" not in script
        assert "ic_history" not in script

    def test_the_script_says_how_often_it_runs(self):
        script = Path("scripts/ashare-retrain.sh").read_text()

        assert "Weekly" in script.splitlines()[1] or "weekly" in script.splitlines()[1]


class TestRetrainGatePicksTheNewestByTrainDate:
    """The incumbent must be chosen by train_date, not by name or file time.

    Version sort ranks w115.pt (trained 2026-01-23) above w12.pt (2026-06-30),
    and a copy leaves w9.pt newer on disk than w12.pt. Both mistakes make the
    gate compare this run against a model from another year and shelve it --
    which is what happened on 2026-11-08, twice, with exit code 0 both times.
    """

    RETRAIN = Path(SCRIPT).resolve().parent / "ashare-retrain.sh"

    def _run_selection(self, tmp_path, files):
        import json
        import os
        import re
        import subprocess

        text = self.RETRAIN.read_text()
        fn = re.search(r"_wf_train_date\(\) \{.*?\n\}", text, re.S)
        block = re.search(r"CUR_MODEL=\$\(.*?\n\)", text, re.S)
        assert fn and block

        models = tmp_path / "models"
        models.mkdir()
        for name, (train_date, mtime) in files.items():
            (models / name).write_bytes(b"x")
            os.utime(models / name, (mtime, mtime))
            if train_date:
                (models / f"{name[:-3]}.meta.json").write_text(
                    json.dumps({"model_file": name, "train_date": train_date})
                )

        harness = tmp_path / "sel.sh"
        harness.write_text(
            "#!/bin/bash\n"
            f"PROJECT_DIR={tmp_path}\n"
            f"{fn.group(0)}\n{block.group(0)}\n"
            'echo "CUR_MODEL=$CUR_MODEL"\n'
        )
        out = subprocess.run(["bash", str(harness)], capture_output=True, text=True)
        return out.stdout.strip().split("=", 1)[1]

    def test_a_bigger_number_with_an_older_date_is_not_the_incumbent(self, tmp_path):
        chosen = self._run_selection(
            tmp_path,
            {
                "w115.pt": ("2026-01-23", 1_700_000_000),
                "w12.pt": ("2026-06-30", 1_600_000_000),
                "w9.pt": ("2026-03-31", 1_800_000_000),
            },
        )
        assert chosen == "w12.pt", f"picked {chosen}: the gate would shelve every run"

    def test_a_checkpoint_without_a_date_is_not_the_incumbent(self, tmp_path):
        chosen = self._run_selection(
            tmp_path,
            {"w115.pt": ("", 1_800_000_000), "w12.pt": ("2026-06-30", 1_700_000_000)},
        )
        assert chosen == "w12.pt"
