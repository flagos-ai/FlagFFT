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
