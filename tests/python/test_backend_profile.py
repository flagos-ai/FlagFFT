import unittest
from flagfft_codegen.backend_profile import BackendProfile, set_profile, reset_profile
from flagfft_codegen.kernels_common import LeafPlan, contiguous_batch_pack_for


class ProfileTest(unittest.TestCase):
    def profile(self, **overrides):
        device = dict(backend="ix", device_arch="ivcore11", warp_size=64,
                      max_threads_per_block=4096, max_dynamic_shared_memory=65536)
        device.update(overrides)
        return BackendProfile.from_device(device)

    def test_native_launch(self):
        p = self.profile()
        self.assertEqual([p.warps_for(n) for n in (32, 64, 65, 128, 129)], [1, 1, 2, 2, 4])
        self.assertEqual(p.shared_budget(128 * 1024), 65536)
        with self.assertRaises(ValueError):
            p.validate(3)

    def test_invalid_device(self):
        for field, value in (("warp_size", None), ("max_threads_per_block", 16),
                             ("max_dynamic_shared_memory", -1)):
            with self.assertRaises(ValueError):
                self.profile(**{field: value})

    def test_cache_isolation(self):
        from dataclasses import replace
        p = self.profile()
        self.assertNotEqual(p.fingerprint, replace(p, policy="legacy").fingerprint)
        self.assertNotEqual(p.fingerprint, replace(p, max_dynamic_shared_memory=32768).fingerprint)

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


if __name__ == "__main__":
    unittest.main()
