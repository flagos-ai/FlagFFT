# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Compatibility facade for the codegen CLI and generated-source helpers.

The native runtime invokes ``python -m flagfft_codegen.jit_source``; this
module keeps that entry point while the implementation lives in
``emit.py`` / ``metadata.py`` / ``cli.py``.
"""

from .cli import main
from .emit import *  # noqa: F401,F403
from .metadata import *  # noqa: F401,F403

__all__ = [
    "_BLUESTEIN_BLOCK",
    "_BLUESTEIN_NUM_STAGES",
    "_BLUESTEIN_NUM_WARPS",
    "_RADER_BLOCK",
    "_RADER_NUM_STAGES",
    "_RADER_NUM_WARPS",
    "_RESHAPE_NUM_STAGES",
    "_RESHAPE_NUM_WARPS",
    "_arg_signature",
    "_bluestein_kernel_source",
    "_csv_ints",
    "_dtype_suffix",
    "_emit_bluestein_jit_kernel",
    "_emit_r2c_pointwise_jit_kernel",
    "_emit_rader_jit_kernel",
    "_emit_reshape_jit_kernel",
    "_emit_tiled_transpose3d_jit_kernel",
    "_emit_tiled_transpose_jit_kernel",
    "_metadata",
    "_module_source",
    "_pointer_signature",
    "_rader_kernel_source",
    "_signature",
    "_transpose3d_v2_supported",
    "_zero_other",
    "emit_jit_kernel",
    "main",
]

if __name__ == "__main__":
    main()
