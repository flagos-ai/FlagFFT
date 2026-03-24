"""
Triton codelet implementations for FFT.

This module provides Triton-based implementations of radix-N FFT butterfly operations.
Each radix-N module provides:
- batch_fwd_radN_b1: Batch forward FFT
- batch_inv_radN_b1: Batch inverse FFT
- triton_fwd_radN_b1: Alias for batch_fwd_radN_b1
- triton_inv_radN_b1: Alias for batch_inv_radN_b1
"""

# Radix-2
from .radix_2 import (
    fwd_rad2_b1,
    inv_rad2_b1,
    triton_fwd_rad2_b1,
    triton_inv_rad2_b1,
    batch_fwd_rad2_b1,
    batch_inv_rad2_b1,
)

# Radix-3
from .radix_3 import (
    triton_fwd_rad3_b1,
    triton_inv_rad3_b1,
    batch_fwd_rad3_b1,
    batch_inv_rad3_b1,
)

# Radix-4
from .radix_4 import (
    triton_fwd_rad4_b1,
    triton_inv_rad4_b1,
    batch_fwd_rad4_b1,
    batch_inv_rad4_b1,
)

# Radix-5
from .radix_5 import (
    triton_fwd_rad5_b1,
    triton_inv_rad5_b1,
    batch_fwd_rad5_b1,
    batch_inv_rad5_b1,
)

# Radix-6
from .radix_6 import (
    triton_fwd_rad6_b1,
    triton_inv_rad6_b1,
    batch_fwd_rad6_b1,
    batch_inv_rad6_b1,
)

# Radix-7
from .radix_7 import (
    triton_fwd_rad7_b1,
    triton_inv_rad7_b1,
    batch_fwd_rad7_b1,
    batch_inv_rad7_b1,
)

# Radix-8
from .radix_8 import (
    triton_fwd_rad8_b1,
    triton_inv_rad8_b1,
    batch_fwd_rad8_b1,
    batch_inv_rad8_b1,
)

# Radix-9
from .radix_9 import (
    triton_fwd_rad9_b1,
    triton_inv_rad9_b1,
    batch_fwd_rad9_b1,
    batch_inv_rad9_b1,
)

# Radix-10
from .radix_10 import (
    triton_fwd_rad10_b1,
    triton_inv_rad10_b1,
    batch_fwd_rad10_b1,
    batch_inv_rad10_b1,
)

# Radix-11
from .radix_11 import (
    triton_fwd_rad11_b1,
    triton_inv_rad11_b1,
    batch_fwd_rad11_b1,
    batch_inv_rad11_b1,
)

# Radix-13
from .radix_13 import (
    triton_fwd_rad13_b1,
    triton_inv_rad13_b1,
    batch_fwd_rad13_b1,
    batch_inv_rad13_b1,
)

# Radix-16
from .radix_16 import (
    triton_fwd_rad16_b1,
    triton_inv_rad16_b1,
    batch_fwd_rad16_b1,
    batch_inv_rad16_b1,
)

# Radix-17
from .radix_17 import (
    triton_fwd_rad17_b1,
    triton_inv_rad17_b1,
    batch_fwd_rad17_b1,
    batch_inv_rad17_b1,
)

__all__ = [
    # Radix-2
    'fwd_rad2_b1',
    'inv_rad2_b1',
    'triton_fwd_rad2_b1',
    'triton_inv_rad2_b1',
    'batch_fwd_rad2_b1',
    'batch_inv_rad2_b1',
    # Radix-3
    'triton_fwd_rad3_b1',
    'triton_inv_rad3_b1',
    'batch_fwd_rad3_b1',
    'batch_inv_rad3_b1',
    # Radix-4
    'triton_fwd_rad4_b1',
    'triton_inv_rad4_b1',
    'batch_fwd_rad4_b1',
    'batch_inv_rad4_b1',
    # Radix-5
    'triton_fwd_rad5_b1',
    'triton_inv_rad5_b1',
    'batch_fwd_rad5_b1',
    'batch_inv_rad5_b1',
    # Radix-6
    'triton_fwd_rad6_b1',
    'triton_inv_rad6_b1',
    'batch_fwd_rad6_b1',
    'batch_inv_rad6_b1',
    # Radix-7
    'triton_fwd_rad7_b1',
    'triton_inv_rad7_b1',
    'batch_fwd_rad7_b1',
    'batch_inv_rad7_b1',
    # Radix-8
    'triton_fwd_rad8_b1',
    'triton_inv_rad8_b1',
    'batch_fwd_rad8_b1',
    'batch_inv_rad8_b1',
    # Radix-9
    'triton_fwd_rad9_b1',
    'triton_inv_rad9_b1',
    'batch_fwd_rad9_b1',
    'batch_inv_rad9_b1',
    # Radix-10
    'triton_fwd_rad10_b1',
    'triton_inv_rad10_b1',
    'batch_fwd_rad10_b1',
    'batch_inv_rad10_b1',
    # Radix-11
    'triton_fwd_rad11_b1',
    'triton_inv_rad11_b1',
    'batch_fwd_rad11_b1',
    'batch_inv_rad11_b1',
    # Radix-13
    'triton_fwd_rad13_b1',
    'triton_inv_rad13_b1',
    'batch_fwd_rad13_b1',
    'batch_inv_rad13_b1',
    # Radix-16
    'triton_fwd_rad16_b1',
    'triton_inv_rad16_b1',
    'batch_fwd_rad16_b1',
    'batch_inv_rad16_b1',
    # Radix-17
    'triton_fwd_rad17_b1',
    'triton_inv_rad17_b1',
    'batch_fwd_rad17_b1',
    'batch_inv_rad17_b1',
]
