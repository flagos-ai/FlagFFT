from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _register_local_package(name: str, init_path: Path, *, execute: bool) -> None:
    spec = importlib.util.spec_from_file_location(
        name, init_path, submodule_search_locations=[str(init_path.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load local package {name} from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if execute:
        spec.loader.exec_module(module)


def _load_local_module(name: str, module_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load local module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


sys.path.insert(0, str(_PROJECT_ROOT))
for _name in list(sys.modules):
    if _name == "src" or _name.startswith("src."):
        del sys.modules[_name]
_register_local_package("src", _PROJECT_ROOT / "src" / "__init__.py", execute=True)
_register_local_package(
    "src.codegen", _PROJECT_ROOT / "src" / "codegen" / "__init__.py", execute=False
)
_load_local_module("src.codegen.kernels", _PROJECT_ROOT / "src" / "codegen" / "kernels.py")

_SPEC = importlib.util.spec_from_file_location(
    "src.codegen.triton_aot", _PROJECT_ROOT / "src" / "codegen" / "triton_aot.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("failed to load repo-local src.codegen.triton_aot")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["src.codegen.triton_aot"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


if __name__ == "__main__":
    main()
