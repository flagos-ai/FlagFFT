# CTest-Pytest 集成设计

## 目标

将 ctest 测试可执行文件对接到 pytest 中，使 C++ 正确性测试通过 pytest 框架统一运行和报告。

## 集成方式

- pytest 通过子进程直接调用 ctest 测试可执行文件（非 `ctest` 命令）
- 每个 ctest 可执行文件映射为一个 pytest 用例（共 9 个）
- 通过 `--ctest-build-dir` 参数指定构建目录，默认 `build/ctest`

## 测试文件

新增 `tests/ctest/` 目录，包含：

### `tests/ctest/conftest.py`

- 提供 `--ctest-build-dir` pytest 命令行选项，默认值为项目根目录下的 `build/ctest`
- 提供 `ctest_build_dir` fixture（session scope）
- 提供 `run_ctest` fixture（function scope），封装 `subprocess.run()` 调用
- 支持 `FLAGFFT_TEST_PROFILE` 环境变量，允许按 profile 过滤（如 `smoke`）
- 构建目录不存在时 skip 所有测试
- 可执行文件不存在时 skip 对应测试

### `tests/ctest/test_correctness.py`

- 定义 `CTEST_TARGETS` 列表：`test_plan`, `test_exec_c2c`, `test_exec_z2z`, `test_exec_r2c`, `test_exec_c2r`, `test_exec_d2z`, `test_exec_z2d`, `test_exec_r2c_c2r`, `test_exec_d2z_z2d`
- 使用 `pytest.mark.parametrize` 对每个 target 生成一个用例
- 每个用例：运行可执行文件，检查返回码为 0
- 失败时捕获 stdout/stderr 供 pytest 展示

## 运行方式

```sh
# 完整运行
pytest tests/ctest/ -v

# 指定构建目录
pytest tests/ctest/ -v --ctest-build-dir /path/to/build/ctest

# smoke 测试
FLAGFFT_TEST_PROFILE=smoke pytest tests/ctest/ -v
```

## 范围外

- 不在 Python 中实现 Google Test 级别的细粒度参数化（smoke/full 等 profile 仍由 `FLAGFFT_TEST_PROFILE` 环境变量控制在 C++ 侧）
- 不修改 `tests/cli/` 和 `benchmark/` 的现有测试
- 不合并 `tests/conftest.py` 和 `benchmark/conftest.py` 中的重复代码（独立改进）
