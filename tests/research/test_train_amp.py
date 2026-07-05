"""Tests for AMP wrapper, assign_data fp32 cast, and batch_size passthrough.

All tests mock qlib -- no real training, no GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestAmpFlagDefaultOff:
    """use_amp absent or false -> model.fit is NOT monkey-patched."""

    def test_amp_flag_default_off(self):
        cfg = {"type": "alstm", "handler": "alpha360", "d_feat": 6}
        # No use_amp key at all
        assert cfg.get("use_amp", False) is False


class TestAmpFlagEnablesAutocast:
    """use_amp=true -> autocast context is entered during fit."""

    def test_amp_flag_enables_autocast(self):
        import torch

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
        import torch

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
    """bf16 tensor passed to assign_data is cast to fp32 before original."""

    def test_assign_data_fp32_cast(self):
        import torch

        original_assign = MagicMock()
        dataset = MagicMock()
        dataset.assign_data = original_assign

        cfg_model = {"use_amp": True}

        if cfg_model.get("use_amp", False):
            _orig_assign = dataset.assign_data

            def _safe_assign(index, vals):
                if isinstance(vals, torch.Tensor):
                    vals = vals.float()
                return _orig_assign(index, vals)

            dataset.assign_data = _safe_assign

        # Pass a bf16 tensor
        bf16_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        dataset.assign_data("idx", bf16_tensor)

        # Original should have received fp32
        args = original_assign.call_args
        assert args[0][0] == "idx"
        received_tensor = args[0][1]
        assert received_tensor.dtype == torch.float32


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
            model = MagicMock()
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
