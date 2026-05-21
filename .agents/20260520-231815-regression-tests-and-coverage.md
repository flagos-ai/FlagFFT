时间日期: 2026-05-20 23:18:15 Asia/Shanghai

用户指令:
- 检查并修复回归测试错误。
- 审查现有测试是否覆盖当前功能路径，缺失则补充测试，并确保全部检查通过。

工作计划:
- 运行现有 CTest 和 pytest，定位回归失败。
- 修复 Python codegen 测试导入仓库内源码时被旧 site-packages 影子包污染的问题。
- 审查 C API 与 codegen 路径覆盖，补充缺失边界和 generic four-step JIT smoke 测试。
- 更新 README、AGENTS.md、pyproject.toml 中与测试/包容器相关的说明。
- 重建并运行全部回归测试。

落实产物:
- 新增 `src/__init__.py`，作为 Python codegen/tune 的包容器，避免测试导入旧安装包。
- `src/codegen/jit_source.py` 的 Bluestein/reshape JIT emit helper 增加默认 `complex64` dtype，保持直接调用兼容。
- `tests/python/test_codegen.py` 补充 fp64 inner-pack 规则、generic `reshape_pack` / `twiddle_reshape_pack` JIT metadata/source 生成测试，并修正 dtype 后缀文件名断言。
- `tests/ctest/flagfft_c_api_test.cpp` 补充 `flagfftPlan2d` / `flagfftPlan3d` unsupported rank 断言和 C2C in-place unsupported 断言。
- `README.md` 更新 `src/tune/` 目录说明和当前测试覆盖范围。
- `pyproject.toml` 将 `triton` 纳入 test/dev extra，匹配 codegen pytest 的实际依赖。

测试结果:
- `pytest -q tests/python`: 7 passed
- `cmake --build build --target flagfft_c_api_tests flagfft_plan_tests`: passed
- `ctest --test-dir build --output-on-failure`: 2/2 tests passed
- `python src/codegen/jit_source.py --kernel reshape_pack --reshape-n1 4 --reshape-n2 8 --out-dir <tmp>`: generated f32 source and metadata
