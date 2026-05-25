import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ai * br + ar * bi
