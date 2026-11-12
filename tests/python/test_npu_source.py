"""Check source-level integer identities against independent scalar semantics."""
import ast
import pytest
from flagfft_codegen.npu_source import vector_remainders


@pytest.mark.parametrize('divisor',[2,3,5,7,13,17,19,256])
def test_constant_remainders_preserve_index_routes(divisor):
    source=f'def route(x):\n    return (x // 3 + 17) % {divisor}, (x % {divisor}) // 2\n'
    original={};rewritten={}
    exec(source,original);out=vector_remainders(source);exec(out,rewritten)
    assert not any(isinstance(n,ast.Mod) for n in ast.walk(ast.parse(out)))
    for index in range(-4096,8193):
        assert original['route'](index)==rewritten['route'](index)


def test_dynamic_and_float_divisors_are_preserved():
    out=vector_remainders('def f(x,d):\n    return x % d, x % 3.5\n')
    assert sum(isinstance(n,ast.Mod) for n in ast.walk(ast.parse(out)))==2
