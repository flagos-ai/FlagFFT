# MACA 同进程 C API 正确性验收

`tests/python/test_maca_c_api.py` 直接调用已经构建的 `libflagfft.so`，以 `Plan1d → SetStream → Exec → Destroy` 执行每个 case。进程内复用生产 C++ KernelCache，减少重复启动和相同 kernel 的重复生成；不采集性能，不改生产执行逻辑，也不替代尚在进行的统一 runner 验收。

默认不运行设备测试，只有 `FLAGFFT_TEST_MACA=1` 才启用。case 来自 `conf/operators.yaml` 和 `conf/test_matrix.yaml` 中的 1D CT/prime single 六 API，采用合法方向与配置 scale。默认 136 个 case；shape=997 时 8 个。直接调用统一 runner 的 `make_input`、`numpy_reference`、`error_stats`、`judged_stats`、`accuracy_limit`，种子、原生输入精度、inverse 不归一化约定和误差阈值均保持一致。

## 隔离与结果

C++ 的磁盘 cache 根由 `/proc/self/exe` 的真实父目录决定，因此需要复制 Python 可执行文件，符号链接不能隔离。测试要求显式 `FLAGFFT_TEST_MACA_EXE_DIR` 与真实目录一致；`TRITON_CACHE_DIR` 必须是其独立子目录，C++ `.flagfft` 不能重定向到其他目录。不能对同一进程切换优化环境，也不能并发使用同一目录。

目录中的 `.flagfft-maca-test-variant.json` 记录 library、codegen 源码哈希、影响生成/执行的环境和使用中的 tuned DB 哈希；已存在未标识的非空 cache 会被拒绝。变体变化应使用新目录。测试不会修改sys.path来覆盖生产codegen；同时核对当前进程和C++ JIT使用的Python对codegen模块的解析路径，并在manifest分别记录生产codegen与测试/oracle来源。shape/API/scale/direction 筛选和输出目录不属于变体，可以在同一缓存上补测不同矩阵。设备由 validation 在进程启动前限制为单物理卡并持有既有 GPU slot 锁；测试本身不会调度 GPU，不支持 pytest-xdist。

必须显式指定 `FLAGFFT_TEST_MACA_OUTPUT_DIR`，位于结果根的独立套件目录。默认结果根是源码 worktree 上一级的 `results/`，远端挂载使用 `FLAGFFT_TEST_MACA_RESULTS_ROOT=/workspace/FlagFFT-results`。每个 case 保存独立 JSON：输入 seed/scale/dtype 与 SHA256、输出 SHA256、完整 plan、原始及判定后的误差指标、阈值、状态和整个 case 墙钟。此墙钟不代表 kernel 性能。输出保护区另做越界检查。结果目录不能覆盖旧 manifest/case。

## validation 后续小规模交叉核对

以下命令仅供 validation 在现有 GPU 队列允许时运行。`FFT_CODEGEN_SRC` 是冻结的生产codegen源码，`FFT_TEST_SRC` 是本测试及统一runner源码（可以来自另一已提交worktree），`FFT_LIBRARY` 是已构建的 MACA libflagfft.so，`FFT_EXE_DIR` 是当前变体全新的目录，`FFT_RESULTS` 是新的根结果套件目录。保留验证候选所需的 `FLAGFFT_MACA_*` 设置；不要复用其他变体的 cache。

```sh
mkdir -p "$FFT_EXE_DIR"
cp -L /opt/conda/bin/python "$FFT_EXE_DIR/python"
env PYTHONHOME=/opt/conda PYTHONPATH="$FFT_CODEGEN_SRC/python" \
  CUDA_VISIBLE_DEVICES=4 MACA_VISIBLE_DEVICES=4 MC_VISIBLE_DEVICES=4 \
  FLAGFFT_TEST_MACA=1 FLAGFFT_TEST_LIBRARY="$FFT_LIBRARY" \
  FLAGFFT_TEST_MACA_EXE_DIR="$FFT_EXE_DIR" \
  TRITON_CACHE_DIR="$FFT_EXE_DIR/triton-cache" \
  FLAGFFT_TEST_MACA_RESULTS_ROOT=/workspace/FlagFFT-results \
  FLAGFFT_TEST_MACA_OUTPUT_DIR="$FFT_RESULTS/c_api" \
  FLAGFFT_TEST_MACA_SHAPES=997 FLAGFFT_TEST_MACA_APIS=c2c,r2c,c2r \
  FLAGFFT_TEST_MACA_SCALES=1 FLAGFFT_TUNE_DISABLE=1 \
  FLAGFFT_PYTHON=/opt/conda/bin/python \
  "$FFT_EXE_DIR/python" -m pytest -q "$FFT_TEST_SRC/tests/python/test_maca_c_api.py"
```

首轮这 4 个 FP32 case 对照相同 source/variant 的统一 runner `--accuracy-only --ops 1d_prime_single_c2c,1d_prime_single_r2c,1d_prime_single_c2r --shapes 997 --scales 1`。核对 case_id、seed/input_sha256、plan 路由、输出哈希及误差判定；若输出哈希不同，先核对流/执行配置并分析误差，不能只看 pytest PASS 就视为一致。统一 runner 仍使用其独立 capture executable/cache。

小规模交叉核对通过后，再将 API 筛选去掉或改为 `z2z,d2z,z2d` 验收真实 FP64 production 路由；不能拿固定2048的 leaf测试代替997默认2000路由。shape/API 都可用逗号列表；`FLAGFFT_TEST_MACA_DIRECTIONS=forward` 或 `inverse` 只保留合法方向；`FLAGFFT_TEST_MACA_SCALES=all` 复用统一 runner 三种幅度。筛选为空或非法值时 collection 失败，避免误报空矩阵成功。

CPU验证证据位于 `results/20260921_161103_maca_single_c_api_cpu/`：默认case集合与统一runner逐项相同；997六API合法方向的seed/输入hash已记录，C2C forward输入hash与既有统一capture结果逐字节一致；各筛选计数、默认跳过和隔离目录拒绝检查完成。CPU collection不导入Torch或初始化设备。设备执行与统一capture交叉核对尚待validation完成。
