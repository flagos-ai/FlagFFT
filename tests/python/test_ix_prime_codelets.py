"""Compare the emitted arithmetic with a direct DFT, independent of the GPU SDK."""
import cmath
import math
import random
import pytest
from flagfft_codegen.ix_prime_codelets import emit_prime_codelet

@pytest.mark.parametrize('prime',[13,17])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_register_rader_matches_direct_dft(prime,direction):
    rng=random.Random(1729)
    vectors=[[complex(rng.uniform(-1,1),rng.uniform(-1,1)) for _ in range(prime)],
             [complex(j==0) for j in range(prime)],
             [complex(1,-2) for _ in range(prime)]]
    source='\n'.join(emit_prime_codelet('',prime,direction))
    code=compile(source,'<register-rader>','exec')
    sign=-1 if direction=='forward' else 1
    for values in vectors:
        env={name+str(j):getattr(z,component) for j,z in enumerate(values)
             for name,component in (('r','real'),('i','imag'))}
        exec(code,{},env)
        expected=[sum(z*cmath.exp(sign*2j*math.pi*k*j/prime) for j,z in enumerate(values))
                  for k in range(prime)]
        actual=[complex(env['r'+str(j)],env['i'+str(j)]) for j in range(prime)]
        assert max(abs(a-b) for a,b in zip(actual,expected))<1e-11


def test_c550_four_step_leaf_can_select_register_rader(monkeypatch):
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io

    plan = LeafPlan(1768, (17, 13, 8), 1, 1, 4, (), 2048)
    profile = set_profile(BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
        "max_dynamic_shared_memory": 65536,
    }))
    try:
        monkeypatch.setenv("FLAGFFT_MACA_RADER_CODELET", "0")
        _, baseline = _build_leaf_kernel_source_for_io(
            plan, io_mode="four_step_col", four_step_n1=375, four_step_n2=1768,
        )
        monkeypatch.setenv("FLAGFFT_MACA_RADER_CODELET", "1")
        _, candidate = _build_leaf_kernel_source_for_io(
            plan, io_mode="four_step_col", four_step_n1=375, four_step_n2=1768,
        )
        assert "= _fwd_rad17_b1(" in baseline
        assert "= _fwd_rad13_b1(" in baseline
        assert "= _fwd_rad17_b1(" not in candidate
        assert "= _fwd_rad13_b1(" not in candidate
        assert "ixr_product0_r" in candidate
    finally:
        reset_profile(profile)
