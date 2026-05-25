import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def _cmul(ar, ai, br, bi):
  return ar * br - ai * bi, ai * br + ar * bi
