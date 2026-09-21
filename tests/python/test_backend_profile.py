import unittest
from unittest.mock import patch

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan, contiguous_batch_pack_for


class ProfileTest(unittest.TestCase):
    def test_bluestein_boundary_packing_matches_generated_batch_stride(self):
        from pathlib import Path

        from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
        from flagfft_codegen.metadata import _metadata
        from flagfft_codegen.target import set_codegen_target

        set_codegen_target("maca:80:64")
        try:
            for n, factors, lanes in (
                (16, (16,), 1),
                (32, (32,), 1),
                (256, (16, 16), 16),
                (2048, (16, 16, 8), 128),
            ):
                plan = LeafPlan(n, factors, 1, lanes, 2, (), n)
                for kind, mode in (
                    ("leaf_bluestein_prepare", "bluestein_prepare_leaf"),
                    ("leaf_bluestein_finish", "bluestein_finish_leaf"),
                ):
                    with self.subTest(n=n, kind=kind):
                        name, source = _build_leaf_kernel_source_for_io(
                            plan, io_mode=mode, prime_n=n // 2 - 1
                        )
                        metadata = _metadata(
                            module_path=Path("unused.py"), kernel_name=name,
                            arg_names=[], plan=plan, kernel_type=kind,
                            n1=0, n2=0, dtype=plan.dtype,
                        )
                        self.assertIn(
                            f"batch_id = pid * {metadata['batch_per_block']}", source
                        )
                        self.assertEqual(
                            metadata["batch_per_block"], contiguous_batch_pack_for(plan)
                        )
        finally:
            set_codegen_target("")

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
        with self.assertRaises(ValueError):
            self.profile(backend="unknown")
        with self.assertRaises(ValueError):
            self.profile(device_arch="")

    def test_missing_launch_facts_fall_back_to_backend_defaults(self):
        for overrides in (
            {"warp_size": None},
            {"max_threads_per_block": 16},
            {"warp_size": 128},
        ):
            profile = self.profile(**overrides)
            self.assertEqual(profile.warp_size, 64)
            self.assertEqual(profile.max_threads_per_block, 4096)
            self.assertEqual(profile.facts_source, "backend_default")
        dropped = self.profile(max_dynamic_shared_memory=-1)
        self.assertIsNone(dropped.max_dynamic_shared_memory)
        self.assertEqual(dropped.facts_source, "backend_default")

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
        # Without an explicit profile the packing helper falls back to the
        # installed Triton plugin, so pin a declared non-MACA backend to keep
        # this assertion host-independent.
        default = BackendProfile(backend="cuda", device_arch="80")
        token = set_profile(default)
        try:
            self.assertEqual(contiguous_batch_pack_for(plan), 2)
        finally:
            reset_profile(token)
        token = set_profile(replace(self.profile(), policy="packed"))
        try:
            self.assertEqual(contiguous_batch_pack_for(plan), 4)
        finally:
            reset_profile(token)
        token = set_profile(default)
        try:
            self.assertEqual(contiguous_batch_pack_for(plan), 2)
        finally:
            reset_profile(token)

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

    def test_single_smem_buffer_only_for_large_single_group_leaves(self):
        from flagfft_codegen.kernels_common import _use_single_smem_buffer

        token = set_profile(self.profile(max_dynamic_shared_memory=131072))
        try:
            large = LeafPlan(2048, (16, 16, 8), 1, 128, 4, (), 2048)
            multi_group = LeafPlan(1024, (8, 8, 4, 4), 1, 128, 4, (), 1024)
            small = LeafPlan(256, (16, 16), 1, 128, 4, (), 256)
            self.assertTrue(_use_single_smem_buffer(large, io_mode="contiguous"))
            self.assertFalse(_use_single_smem_buffer(multi_group, io_mode="contiguous"))
            self.assertFalse(_use_single_smem_buffer(small, io_mode="contiguous"))
            self.assertFalse(_use_single_smem_buffer(large, io_mode="four_step_row"))
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

    def test_maca_four_step_packing_respects_runtime_smem_limit(self):
        from flagfft_codegen.kernels_common import four_step_col_inner_pack_for

        profile = BackendProfile.from_device(
            {
                "backend": "maca",
                "device_arch": "102",
                "warp_size": 64,
                "max_threads_per_block": 1024,
                # The driver-reported opt-in value is intentionally not enough
                # to make this kernel launchable on C550.
                "max_dynamic_shared_memory": 131072,
            },
            "legacy",
        )
        token = set_profile(profile)
        try:
            # n=663000 selects a 1768-point column leaf. Pack four allocates
            # 4 * 8192 float32 values = 128 KiB; pack two is the largest safe
            # choice at the 64 KiB runtime limit.
            oversized = LeafPlan(1768, (17, 13, 8), 1, 1, 4, (), 2048)
            self.assertEqual(
                four_step_col_inner_pack_for(375, 1768, "complex64", oversized), 2
            )

            safe = LeafPlan(1024, (16, 8, 8), 1, 64, 2, (), 1024)
            self.assertEqual(
                four_step_col_inner_pack_for(1024, 1024, "complex64", safe), 4
            )

            # All-power-of-two direct exchange is register-routed.  Its
            # generated C550 kernel remains launchable at pack eight even
            # though the conservative generic four-buffer estimate would
            # reject that pack at the 64 KiB runtime limit.
            with patch.dict(
                "os.environ",
                {
                    "FLAGFFT_MACA_EXCHANGE": "direct_all",
                    "FLAGFFT_MACA_INNER_PACK": "8",
                },
            ):
                self.assertEqual(
                    four_step_col_inner_pack_for(1024, 1024, "complex64", safe),
                    8,
                )

            # The same bypass is not safe for FP64: an 8-slot [16,16,8]
            # column leaf requests 128 KiB on C550 and must remain capped.
            double_plan = LeafPlan(
                2048,
                (16, 16, 8),
                1,
                128,
                4,
                (),
                2048,
                dtype="complex128",
            )
            with patch.dict(
                "os.environ",
                {
                    "FLAGFFT_MACA_EXCHANGE": "direct",
                    "FLAGFFT_MACA_INNER_PACK": "8",
                },
            ):
                self.assertEqual(
                    four_step_col_inner_pack_for(
                        512, 2048, "complex128", double_plan
                    ),
                    1,
                )
        finally:
            reset_profile(token)

    def test_maca_direct_all_mixed_fp64_pack_respects_padded_radix(self):
        from flagfft_codegen.kernels_common import four_step_col_inner_pack_for

        profile = BackendProfile.from_device(
            {
                "backend": "maca",
                "device_arch": "102",
                "warp_size": 64,
                "max_threads_per_block": 1024,
                # Keep the test independent of the optimistic device report.
                "max_dynamic_shared_memory": 131072,
            },
            "legacy",
        )
        token = set_profile(profile)
        try:
            # 185640 = 390 * 476; the column leaf is [17, 7, 4]. With
            # direct_all, radix 17 is joined as radix 32. FP64 P4 therefore
            # requests 128 KiB (128 lanes * 4 slots * 32 * 8 bytes), while P2
            # fits the 64 KiB C550 launch limit.
            plan = LeafPlan(
                476,
                (17, 7, 4),
                1,
                1,
                2,
                (),
                512,
                dtype="complex128",
            )
            with patch.dict(
                "os.environ",
                {
                    "FLAGFFT_MACA_EXCHANGE": "direct_all",
                    "FLAGFFT_MACA_INNER_PACK": "4",
                },
            ):
                self.assertEqual(
                    four_step_col_inner_pack_for(
                        390, 476, "complex128", plan
                    ),
                    2,
                )

            # The same layout is exactly 64 KiB per joined component in FP32,
            # so the adaptive path should retain P4 there.
            fp32_plan = LeafPlan(
                476,
                (17, 7, 4),
                1,
                1,
                2,
                (),
                512,
                dtype="complex64",
            )
            with patch.dict(
                "os.environ",
                {
                    "FLAGFFT_MACA_EXCHANGE": "direct_all",
                    "FLAGFFT_MACA_INNER_PACK": "4",
                },
            ):
                self.assertEqual(
                    four_step_col_inner_pack_for(
                        390, 476, "complex64", fp32_plan
                    ),
                    4,
                )
        finally:
            reset_profile(token)

    def test_maca_global_p8_is_limited_to_large_fp32_power_of_two_leaves(self):
        from flagfft_codegen.kernels_common import four_step_col_inner_pack_for

        profile = BackendProfile.from_device(
            {
                "backend": "maca",
                "device_arch": "102",
                "warp_size": 64,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 131072,
            },
            "legacy",
        )
        token = set_profile(profile)
        try:
            env = {
                "FLAGFFT_MACA_EXCHANGE": "direct_all",
                "FLAGFFT_MACA_INNER_PACK": "8",
            }
            with patch.dict("os.environ", env):
                large_power2 = LeafPlan(
                    1024, (16, 8, 8), 1, 64, 2, (), 1024, dtype="complex64"
                )
                self.assertEqual(
                    four_step_col_inner_pack_for(
                        1024, 1024, "complex64", large_power2
                    ),
                    8,
                )

                # The mixed 476-point leaf is part of the 185640/large
                # mixed-radix routes.  P8 must not force its padded radix-32
                # join; the derived policy selects P2 here because the
                # padded radix-32 allocation is the limiting resource.
                mixed = LeafPlan(
                    476, (17, 7, 4), 1, 1, 2, (), 512, dtype="complex64"
                )
                self.assertEqual(
                    four_step_col_inner_pack_for(390, 476, "complex64", mixed),
                    2,
                )

                # A short power-of-two leaf has no work to amortize P8's
                # register pressure and should also use the derived pack.
                short = LeafPlan(
                    128, (8, 4, 4), 1, 16, 1, (), 128, dtype="complex64"
                )
                self.assertEqual(
                    four_step_col_inner_pack_for(128, 128, "complex64", short),
                    4,
                )
        finally:
            reset_profile(token)


if __name__ == "__main__":
    unittest.main()
