import unittest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan, contiguous_batch_pack_for


class ProfileTest(unittest.TestCase):
    def profile(self, **overrides):
        device = dict(
            backend="ix",
            device_arch="ivcore11",
            warp_size=64,
            max_threads_per_block=4096,
            max_dynamic_shared_memory=65536,
        )
        device.update(overrides)
        return BackendProfile.from_device(device)

    def test_native_launch(self):
        p = self.profile()
        self.assertEqual(
            [p.warps_for(n) for n in (32, 64, 65, 128, 129)], [1, 1, 2, 2, 4]
        )
        self.assertEqual(p.shared_budget(128 * 1024), 65536)
        with self.assertRaises(ValueError):
            p.validate(3)

    def test_invalid_device(self):
        for field, value in (
            ("warp_size", None),
            ("max_threads_per_block", 16),
            ("max_dynamic_shared_memory", -1),
            ("backend", "unknown"),
        ):
            with self.assertRaises(ValueError):
                self.profile(**{field: value})

    def test_cache_isolation(self):
        from dataclasses import replace

        p = self.profile()
        self.assertNotEqual(p.fingerprint, replace(p, policy="legacy").fingerprint)
        self.assertNotEqual(
            p.fingerprint, replace(p, max_dynamic_shared_memory=32768).fingerprint
        )

    def test_explicit_ix_disables_cuda_specializations(self):
        from flagfft_codegen.kernels_common import (
            _ix_backend_active,
            use_tle_fused_twiddle,
        )

        token = set_profile(self.profile())
        try:
            self.assertTrue(_ix_backend_active())
            self.assertFalse(use_tle_fused_twiddle(1024, 1024))
        finally:
            reset_profile(token)

    def test_metadata_native_warps(self):
        from pathlib import Path

        from flagfft_codegen.metadata import _metadata

        token = set_profile(self.profile())
        try:
            metadata = _metadata(
                module_path=Path("unused.py"),
                kernel_name="unused",
                arg_names=[],
                plan=LeafPlan(1024, (16, 16, 4), 1, 64, 2, (), 1024),
                kernel_type="leaf",
                n1=0,
                n2=0,
                dtype="complex64",
            )
            self.assertEqual(metadata["num_warps"], 1)
        finally:
            reset_profile(token)

    def test_packing_uses_profile_and_restores(self):
        from dataclasses import replace

        plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256)
        self.assertEqual(contiguous_batch_pack_for(plan), 2)
        token = set_profile(replace(self.profile(), policy="packed"))
        try:
            self.assertEqual(contiguous_batch_pack_for(plan), 4)
        finally:
            reset_profile(token)
        self.assertEqual(contiguous_batch_pack_for(plan), 2)

    def test_four_step_respects_device_memory(self):
        from flagfft_codegen.kernels_common import four_step_col_inner_pack_for

        token = set_profile(self.profile(max_dynamic_shared_memory=32768))
        try:
            plan = LeafPlan(1024, (16, 16, 4), 1, 64, 2, (), 1024)
            self.assertEqual(
                four_step_col_inner_pack_for(1024, 1024, "complex64", plan), 1
            )
        finally:
            reset_profile(token)

    def test_live_value_budget_preserves_small_pack_and_bounds_large_pack(self):
        from dataclasses import replace

        token = set_profile(replace(self.profile(), policy="balanced"))
        try:
            small = LeafPlan(256, (16, 16), 1, 16, 1, (), 256)
            large = LeafPlan(1024, (32, 32), 1, 32, 1, (), 1024)
            changing_lanes = LeafPlan(128, (8, 4, 4), 1, 16, 1, (), 128)
            self.assertEqual(contiguous_batch_pack_for(small), 4)
            self.assertEqual(contiguous_batch_pack_for(large), 1)
            self.assertEqual(contiguous_batch_pack_for(changing_lanes), 2)
        finally:
            reset_profile(token)

    def test_four_step_packing_uses_profile_budget(self):
        from dataclasses import replace

        from flagfft_codegen.kernels_common import (
            four_step_col_inner_pack_for,
            four_step_row_inner_pack_for,
        )

        plan = LeafPlan(1024, (32, 32), 1, 32, 2, (), 1024)
        legacy = replace(
            self.profile(max_dynamic_shared_memory=131072), policy="legacy"
        )
        profiled = replace(legacy, policy="balanced")
        token = set_profile(legacy)
        try:
            self.assertEqual(
                four_step_row_inner_pack_for(1024, 1024, "complex64", plan), 1
            )
        finally:
            reset_profile(token)
        token = set_profile(profiled)
        try:
            self.assertGreater(
                four_step_row_inner_pack_for(1024, 1024, "complex64", plan), 1
            )
            self.assertGreater(
                four_step_col_inner_pack_for(1024, 1024, "complex64", plan), 1
            )
        finally:
            reset_profile(token)


if __name__ == "__main__":
    unittest.main()
