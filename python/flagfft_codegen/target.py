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

"""Target capabilities used while emitting a module, without GPU initialization."""

from contextvars import ContextVar

_target = ContextVar("flagfft_codegen_target", default="")


def set_codegen_target(target: str) -> None:
    if target and len(target.split(":")) != 3:
        raise ValueError(f"Invalid Triton target: {target!r}")
    if target and int(target.rsplit(":", 1)[1]) <= 0:
        raise ValueError(f"Invalid warp size in target: {target!r}")
    _target.set(target)


def backend_name() -> str:
    target = _target.get()
    return target.split(":", 1)[0] if target else ""


def warp_size() -> int:
    target = _target.get()
    if target:
        return int(target.rsplit(":", 1)[1])
    try:
        from triton._C import libtriton

        if hasattr(libtriton, "metax"):
            return 64
    except ImportError:
        pass
    return 32
