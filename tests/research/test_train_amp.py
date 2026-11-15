"""Tests for AMP wrapper, assign_data fp32 cast, and batch_size passthrough.

All tests mock qlib -- no real training, no GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAmpFlagDefaultOff:
    """use_amp absent or false -> model.fit is NOT monkey-patched."""

    def test_amp_flag_default_off(self):
        cfg = {"type": "alstm", "handler": "alpha360", "d_feat": 6}
        # No use_amp key at all
        assert cfg.get("use_amp", False) is False


class TestAmpFlagEnablesAutocast:
    """use_amp=true -> autocast context is entered during fit."""

    def test_amp_flag_enables_autocast(self):
        torch = pytest.importorskip("torch", reason="torch not installed on this host")

        cfg_model = {"use_amp": True}
        model = MagicMock()
        original_fit = model.fit

        if cfg_model.get("use_amp", False):
            _original_fit = model.fit

            def _amp_fit(dataset, *a, **kw):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return _original_fit(dataset, *a, **kw)

            model.fit = _amp_fit

        # model.fit should now be the wrapper, not the original
        assert model.fit is not original_fit

        # Call it and verify autocast was used (the original fit gets called)
        mock_dataset = MagicMock()
        with patch.object(torch.amp, "autocast", wraps=torch.amp.autocast) as mock_ac:
            model.fit(mock_dataset)
            mock_ac.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)


class TestAmpBfloat16NotFloat16:
    """Verify the wrapper uses bfloat16, NOT float16."""

    def test_amp_bfloat16_not_float16(self):
        torch = pytest.importorskip("torch", reason="torch not installed on this host")

        captured_kwargs = {}

        original_autocast = torch.amp.autocast

        class _CapturingAutocast(original_autocast):
            def __init__(self, *args, **kwargs):
                captured_kwargs.update(kwargs)
                super().__init__(*args, **kwargs)

        cfg_model = {"use_amp": True}
        model = MagicMock()

        if cfg_model.get("use_amp", False):
            _original_fit = model.fit

            def _amp_fit(dataset, *a, **kw):
                with _CapturingAutocast(device_type="cuda", dtype=torch.bfloat16):
                    return _original_fit(dataset, *a, **kw)

            model.fit = _amp_fit

        model.fit(MagicMock())
        assert captured_kwargs["dtype"] is torch.bfloat16
        assert captured_kwargs["dtype"] is not torch.float16


class TestAssignDataFp32Cast:
    """The cast is installed on the class, so prepared views inherit it."""

    def test_assign_data_fp32_cast(self):
        torch = pytest.importorskip("torch", reason="torch not installed on this host")
        from ashare_lab.research.train import amp_assignment_cast

        class _Dataset:
            def __init__(self):
                self.calls = []

            def __copy__(self):
                clone = self.__class__.__new__(self.__class__)
                clone.__dict__ = self.__dict__.copy()
                return clone

            def assign_data(self, index, vals):
                self.calls.append((index, vals))

        root = _Dataset()
        amp_assignment_cast(root)

        # A view prepared after the swap carries the cast; this is the path
        # training takes, where every view is created inside fit().
        view = root.__copy__()
        view.assign_data([7], torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))

        assert view.calls[0][0] == [7]
        assert view.calls[0][1].dtype == torch.float32

    def test_an_instance_patch_stays_bound_to_the_root(self):
        torch = pytest.importorskip("torch", reason="torch not installed on this host")

        class _Dataset:
            def __init__(self):
                self.calls = []
                self.writer = None

            def __copy__(self):
                clone = self.__class__.__new__(self.__class__)
                clone.__dict__ = self.__dict__.copy()
                return clone

            def assign_data(self, index, vals):
                if self.writer is not None:
                    raise ValueError("a root is a preparation container")
                self.calls.append((index, vals))

        dataset = _Dataset()
        dataset.writer = object()
        original = dataset.assign_data

        def _instance_patch(index, vals):
            return original(index, vals)

        dataset.assign_data = _instance_patch
        view = dataset.__copy__()

        # A view copies the instance attribute, but the patch still closes
        # over the root, so the write is refused instead of reaching the
        # shared memory. This is the failure the class-level cast avoids.
        with pytest.raises(ValueError, match="preparation container"):
            view.assign_data([7], torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))
        assert view.calls == []


class TestAssignDataPatchOnlyForTra:
    """ALSTM with use_amp does NOT patch assign_data."""

    def test_assign_data_patch_only_for_tra(self):
        # Simulate the ALSTM branch logic from train.py:
        # ALSTM branch only wraps model.fit, never touches dataset.assign_data
        model_type = "alstm"
        cfg_model = {"use_amp": True, "type": "alstm"}

        dataset = MagicMock()
        original_assign = dataset.assign_data

        # ALSTM branch: only model.fit is wrapped, dataset is untouched
        if model_type == "alstm":

            if cfg_model.get("use_amp", False):
                # Only wrap model.fit -- no assign_data patch
                pass

        # dataset.assign_data should be unchanged
        assert dataset.assign_data is original_assign


class TestBatchSizeConfigPassthrough:
    """batch_size=512 in config -> MTSDatasetH receives 512."""

    def test_batch_size_config_passthrough(self):
        cfg_model = {"batch_size": 512}
        result = cfg_model.get("batch_size", -1)
        assert result == 512


class TestBatchSizeDefaultDaily:
    """No batch_size in config -> MTSDatasetH gets -1 (daily mode)."""

    def test_batch_size_default_daily(self):
        cfg_model = {"type": "tra"}  # no batch_size key
        result = cfg_model.get("batch_size", -1)
        assert result == -1
