"""Device facts and versioned code-generation policy (no device import required)."""
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class BackendProfile:
    backend: str = "cuda"
    device_arch: str = "unspecified"
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_dynamic_shared_memory: int | None = None
    policy: str = "legacy"
    policy_version: int = 1
    toolchain: str = "unspecified"
    source_fingerprint: str = "unspecified"
    # Policy budget for live FFT values, not a queried physical register count.
    leaf_live_bytes_per_thread: int = 128

    @classmethod
    def from_device(cls, device: dict, policy: str = "native"):
        if device.get("backend") not in {"cuda", "musa", "ppu", "ix", "maca"} or not device.get("device_arch"):
            raise ValueError("missing or unsupported device identity")
        if policy not in {"legacy", "native", "packed", "balanced"}:
            raise ValueError(f"unknown execution policy: {policy}")
        warp = device.get("warp_size")
        threads = device.get("max_threads_per_block")
        if warp not in (32, 64) or not isinstance(threads, int) or threads < warp:
            raise ValueError("missing or invalid device launch limits")
        shared = device.get("max_dynamic_shared_memory") or device.get("shared_memory_per_block")
        if shared is not None and (not isinstance(shared, int) or shared <= 0):
            raise ValueError("invalid shared-memory limit")
        return cls(device["backend"], device["device_arch"], warp, threads, shared, policy)

    @property
    def lane_width(self):
        return 32 if self.policy == "legacy" else self.warp_size

    @property
    def leaf_target_threads(self):
        return self.warp_size if self.policy in {"packed", "balanced"} else 32

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]

    def shared_budget(self, preferred):
        if self.policy == "legacy" or self.max_dynamic_shared_memory is None:
            return preferred
        return min(preferred, self.max_dynamic_shared_memory)

    def warps_for(self, logical_lanes):
        wanted = max(1, (logical_lanes + self.lane_width - 1) // self.lane_width)
        candidates = [n for n in (1, 2, 4, 8) if n * self.warp_size <= self.max_threads_per_block]
        # Logical lanes may span multiple elements per physical thread.
        return next((n for n in candidates if n >= wanted), candidates[-1])

    def planner_warps(self, hint):
        # The existing plan format expresses its hint in 32-thread units.
        # Convert that budget once at codegen, keeping serialized plans compatible.
        return self.warps_for(hint * 32)

    def validate(self, num_warps):
        if num_warps not in (1, 2, 4, 8) or num_warps * self.warp_size > self.max_threads_per_block:
            raise ValueError(f"illegal launch: {num_warps} warps on {self}")


_PROFILE = ContextVar("flagfft_backend_profile", default=BackendProfile())


def current_profile():
    return _PROFILE.get()


def set_profile(profile):
    return _PROFILE.set(profile)


def reset_profile(token):
    _PROFILE.reset(token)
