# MACA 1D single 优化与验收工作单

基线：`45eb2a8`，single 指 batch=1；目标 speedup = mcFFT median / FlagFFT median >= 0.8。
历史 2026-09-18 验收仅用于定位问题，不能替代当前提交基线。

## 分工与边界

- `maca_validation`：环境核验、当前基线、唯一 GPU 测量调度者、候选独立验收。
- `maca_exchange`：MACA portable exchange 原型，先覆盖 FP32 1024/2048，再检查 four-step 1048576。
- `maca_prime`：小 prime 两段 leaf 融合，随后评估 four-step 边界融合和算法选择。
- 主 agent：方向选择、源码审阅、集成与验收判定。

各任务使用根目录下独立 `FlagFFT-dev-maca-single-*` worktree。本地主机修改并提交，再同步已提交版本到云端同名 worktree；容器只编译运行。暂不修改 dev 分支。

## 实验纪律

1. MACA 容器 `flagfft-maca-card4-adapt`，仅用物理 GPU 4。所有 GPU 测量由 validation 串行执行，容器锁 `/tmp/flagfft-maca-gpu4.lock`；运行前检查其他占用，不能仅依赖团队锁。
2. 每次运行记录 commit、源码校验、git status、import 路径、SDK/Triton/mcFFT/设备环境与实际 launch metadata。
3. 开关改变 codegen 时明确隔离生成模块、JIT cache 和 plan/tune cache；同一代码缓存不可混用为不同变体。
4. correctness 先于 performance；统一 acceptance runner 对 NumPy 校验。新测量 warmup >= 3、iters >= 30，建议 5/50；候选交错 A/B 至少三轮。
5. 结果统一存放根 `results/<YYYYMMDD_HHMMSS>_maca_single_*`，远端记录回传同一结果目录，保留原始结果与失败证据。
6. 正常 event 计时保留作为验收口径。kernel 诊断、连续提交等额外计时分开报告，不能混用或替代原口径。

## 阶段与停止条件

| 阶段 | 实验 | 判断依据 |
|---|---|---|
| 基线 | C2C 双方向：CT 16/1024/2048/8192/16384/185640/1048576；prime 997/1009/524287 | 当前代码正确性、逐 case mcFFT 对比、实际路径 |
| 交换 | join/静态张量交换对照原 gather+where | 1024/2048 完整 FFT 延迟与编译资源；不能仅用源码 gather 数宣称改善 |
| 复用 | 交换应用于 CT 1048576 与 prime 大卷积 | 端到端收益、两个 four-step pass 时间、输出正确性 |
| prime | 997 两段融合，再 1009 Rader/Bluestein 对照；大 prime 边界融合 | 实际 kernel 数、冷编译时间、steady-state 时间、双方向正确性 |
| 集成 | 胜出候选叠加，对照固定基线 | 独立验收，不简单相乘单项收益 |

若 lowering 不支持或出现静默错误，保留最小复现并修改表达方式；若收益不显著，保留实验结论而不默认启用。编译失败、超时或缺少 GPU 正确性不能称为完成的优化。

## 发布前验收

- 12 个 CT/prime single 算子完整尺寸与方向矩阵：C2C/Z2Z/R2C/C2R/D2Z/Z2D。
- 报告每个算子几何平均、每 case speedup、>=0.8 覆盖率、相对当前基线变化；不能用总平均遮盖严重回退。
- 检查共享代码影响的 batch、2D、3D 代表形状及 CPU/codegen 测试；开关关闭时保持原语义。
- 只有证据充分的配置才考虑默认启用；未通过完整验收的原型保持 opt-in。
- 目标未达到时明确报告实际水平与剩余瓶颈，不把原型交付等同于达成 0.8。
