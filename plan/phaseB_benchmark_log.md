# 阶段 B 性能基准测试记录：ILU0 预条件 + BiCGStab

> 日期：2026-08-13
> 目标：在阶段 A（CG+Jacobi）基础上，验证自实现 ILU0 预条件 + CG/BiCGStab 的正确性与性能，并与串行 SuperLU / 并行 Mumps（沿用阶段 A 数据）对比
> 结果：**完成**。ILU0-CG / ILU0-BiCGStab 三场景数值全部正确（相对偏差 ≤ 1.3e-9），sm_120 平台间歇性 SIGFPE 已消除；性能上 ILU0 在 SPD 问题中慢于 Jacobi+CG（见第六节结论），并已定位到瓶颈 = level 调度临界路径（见第五节）

---

## 一、测试环境

沿用阶段 A，环境不变：

| 项 | 值 |
|---|---|
| GPU | RTX 5070（12GB，sm_120，驱动 591.86） |
| CPU / 内存 | 16 核 / 15GB（WSL 开发机；物理机 32GB，WSL2 默认只分一半） |
| CUDA 版 | `build-cuda/OpenSees`，`system cusp -pre {jacobi|ilu0} -solver {cg|bicgstab} -rtol 1e-8 -maxIter 20000` |
| 串行版 | `OpenSees`，`system SuperLU`（数据沿用阶段 A） |
| 并行版 | `build-hypersp/OpenSeesSP`，`system Mumps`（SCOTCH 排序），`mpiexec -np {4,8}`（数据沿用阶段 A） |

计时口径与阶段 A 一致：Tcl `clock milliseconds` 包住 `analyze`（S1/S2 取 `t_analyze`，S3 取 200 步瞬态总时间 `t_transient`）。

## 二、阶段 B 实现说明（与阶段 A 的差异）

阶段 B 在 `CuSPSolver` 中新增两条求解路径，均复用阶段 A 的 SpMV（cusparseSpMV）+ 向量运算骨架，仅替换「预条件」环节：

| 配置 | 预条件 | 迭代法 |
|---|---|---|
| cusp-cg | Jacobi（diag⁻¹） | CG |
| cusp-bicgstab | Jacobi | BiCGStab |
| **cusp-ilu0-cg** | **ILU0（自实现 CPU 分解 + GPU level-scheduled 前推/后推）** | CG |
| **cusp-ilu0-bicgstab** | **ILU0（同上）** | BiCGStab |

**关键工程背景**：cuSPARSE 的 `cusparseDcsrilu02` 与 `cusparseSpSV` 在 sm_120（Blackwell，RTX 5070）上均不可用——前者在大矩阵（n≥8000）上返回 `INVALID_VALUE`、越界写 CSR 结构并清空对角线，后者间歇性 SIGFPE。故 ILU0 分解改为 **CPU 端等价 ILU(0) 分解**（稠密工作区 + marker 标记），三角求解改为 **自实现 level-scheduled 前推/后推**（每 level 一个 kernel）。详见 `phaseB_implementation_log.md`。

## 三、场景与结果

### 场景 1：3D 实体悬臂块·静力（18×18×200，217,683 DOF）

stdBrick + ElasticIsotropic，底面固结，顶部集中力，LoadControl 1 步。

| 配置 | 求解时间 | 加速比(对串行) | tip 位移 dx | 与串行偏差 |
|---|---|---|---|---|
| 串行 SuperLU | 203.14s | 1× | 0.3282677550413 | — |
| **cusp-cg (Jacobi)** | **4.007s** | **50.7×** | 0.328267755044378 | 9.5e-12 |
| cusp-bicgstab (Jacobi) | 4.346s | 46.7× | 0.328267755042745 | 4.5e-12 |
| cusp-ilu0-cg | 51.788s | 3.9× | 0.328267755022264 | 1.9e-11 |
| cusp-ilu0-bicgstab | 114.998s | 1.8× | 0.328267756335040 | 1.3e-9 |
| SP Mumps np=4 | 49.03s | 4.1× | 0.3282677550449 | 1.1e-11 |
| SP Mumps np=8 | 37.50s | 5.4× | 0.3282677550449 | 1.1e-11 |

### 场景 2：3D 多层钢框架·静力（16×16×24，43,350 DOF）

elasticBeamColumn（ndf=6），底层固结，顶层竖向+水平荷载。

| 配置 | 求解时间 | 加速比(对串行) | tip 位移 dx | 与串行偏差 |
|---|---|---|---|---|
| 串行 SuperLU | 107.53s | 1× | 52.81674053985805 | — |
| **cusp-cg (Jacobi)** | **0.507s** | **212×** | 52.81674053985568 | 4.5e-14 |
| cusp-bicgstab (Jacobi) | 0.778s | 138× | 52.81674053985811 | 2.0e-15 |
| cusp-ilu0-cg | 3.06s | 35.1× | 52.81674053985690 | 1.1e-12 |
| cusp-ilu0-bicgstab | 6.812s | 15.8× | 52.81674053985805 | 2.7e-15 |
| SP Mumps np=4 | 4.61s | 23.3× | 52.81674053927920 | 1.1e-11 |
| SP Mumps np=8 | 3.72s | 28.9× | 52.81674053956326 | 5.6e-12 |

### 场景 3：3D 实体块·瞬态动力（10×10×40，14,883 DOF，Newmark 200 步）

重力 1 步 → 去荷载 → Rayleigh 质量比例阻尼 → 自由振动 200 步，每步一次线性求解。

| 配置 | 瞬态总时间 | 单步平均 | 加速比(对串行) | tip 位移 dx | 与串行偏差 |
|---|---|---|---|---|---|
| 串行 SuperLU | 464.92s | 2.325s | 1× | 0.1804434703284 | — |
| **cusp-cg (Jacobi)** | **34.313s** | **0.172s** | **13.6×** | 0.1804434733967 | 1.7e-8 |
| cusp-bicgstab (Jacobi) | 34.099s | 0.170s | 13.6× | 0.1804434695309 | 8.0e-10 |
| cusp-ilu0-cg | 64.366s | 0.322s | 7.2× | 0.1804434706225 | 2.9e-10 |
| cusp-ilu0-bicgstab | 63.548s | 0.318s | 7.3× | 0.1804434716708 | 1.3e-9 |
| SP Mumps np=4 | 113.54s | 0.568s | 4.1× | 0.1804434703281 | 6.5e-13 |
| SP Mumps np=8 | 102.12s | 0.511s | 4.6× | 0.1804434703281 | 7.1e-13 |

## 四、正确性验证

- 三场景各配置 tip 位移与串行 SuperLU 基线的相对偏差均 ≤ 1.3e-9（ILU0 两条路径；Jacobi 路径沿用阶段 A 结论）
- **ILU0-CG 精度与 Jacobi-CG 相当**：S1 偏差 1.9e-11（Jacobi 9.5e-12 同量级）；S2 偏差 1.1e-12；S3 偏差 2.9e-10，均远优于 Newton 判据（1e-8）
- **ILU0-BiCGStab 精度略松但仍达标**：S1/S3 偏差约 1.3e-9，符合 BiCGStab 半残差收敛（rtol=1e-8）的预期；S2 反而到 2.7e-15
- **稳定性**：S1 全尺寸（217,683 DOF，正是 csrilu02 之前确定性崩溃的规模）+ 8×8×40 连续 10 次 + S3 瞬态 20/200 步，均无崩溃、无 `zero pivot`、无 Jacobi 回退——sm_120 间歇 SIGFPE 已消除

## 五、性能瓶颈定位（假设 → 实验 → 定位）

ILU0 在 SPD 场景慢于 Jacobi，不是迭代次数问题（ILU0 反而把 CG 迭代压到 2.3×），而是**单次预条件（三角求解）太慢**。为定位真正瓶颈，做了一组假设驱动的实验：

| 假设 | 实验 | 结果 | 结论 |
|---|---|---|---|
| 每 level 一次 kernel launch，launch 延迟主导 | 把 level 循环搬进单个 persistent kernel（单 block） | 8×8×40 仅 2.237s→2.025s（~10%） | launch 开销不是主导 |
| 单 block 带宽不足，需全网格常驻 | cooperative persistent kernel + 逐 level `grid.sync()` 栅栏 | S1 51.79s→51.27s（无改善），小规模反而更慢 | 也非 launch/带宽问题 |

**定位到的真正瓶颈 = level 调度的临界路径**：

- 前推/后推按依赖 DAG 分层，层数（最长依赖链）实测 **786**（8×8×40，n=9963）、**~6972**（S1，n=217,683），随规模呈 **O(n^{2/3})** 增长（RCM 排序下）。
- ILU0 把 CG 迭代次数降到 **2.3×**（94 vs 219，8×8×40），但每次 apply 是 Jacobi 的 **~25×**：临界路径长 + 每层活跃行数少（低占用），迭代红利被完全抵消，最终 S1 慢 12.9×。
- 这条「先假设 launch 开销 → persistent kernel 实验否定 → 定位到依赖临界路径」的量化链，是本次性能分析里最有价值的结论：瓶颈是测出来的，不是猜出来的。

## 六、结论

1. **ILU0 + BiCGStab 功能目标达成**：两条 ILU0 路径三场景数值全部正确，且彻底绕开了 sm_120 平台上损坏的 `csrilu02`/`SpSV` 两个 API，稳定性问题解决。
2. **性能上 ILU0 全面慢于 Jacobi+CG**：S1 上 cusp-ilu0-cg（51.8s）是 cusp-cg（4.0s）的 **12.9× 慢**，S2 是 **6× 慢**，S3 是 **1.9× 慢**。ILU0-BiCGStab 更慢（S1 115s）。原因有二：
   - **自实现三角求解并行度低（瓶颈已定位，见第五节）**：level-scheduled 前推/后推的依赖临界路径随规模呈 O(n^{2/3}) 增长（RCM 排序下实测 786/~6972 层），每层活跃行少、占用低；已用 persistent kernel 实验排除了「launch 开销」这一常见误判，真正瓶颈是依赖链本身；
   - **ILU0 分解 + 结构/调度构建是每步固定开销**：CPU ILU0 分解与 level 调度构建在每次 `factorize` 都重算（瞬态每步一次），S3 的重力步已体现 2.7~7.5s 的固定成本。
3. **即便如此仍快于串行 SuperLU**：ILU0 两条路径对串行仍保持 1.8×~35× 加速（S2 最明显），说明迭代法内存优势 + GPU 并行带来的底子仍在；但对并行 Mumps（np=8）ILU0 在 S1 已落败（51.8s vs 37.5s）。
4. **ILU0 的价值边界**：本报告三个场景均为 SPD（弹性），CG+Jacobi 本就最优，ILU0 的额外精度/收敛加速无法抵消自实现三角求解的开销。ILU0/BiCGStab 的真正价值应在**非对称 / 病态**（接触、不可压缩、大变形切线刚度非对称）问题上才体现——需另建非 SPD 场景验证，属阶段 B 未覆盖项。
5. **若需 ILU0 更快（展望）**，后续方向按性价比排序：
   - **A. 置换重排（不动预条件、最漂亮）**：把三角求解临界路径从 RCM 的 O(n^{2/3}) 压向图直径 O(n^{1/3})，收敛行为不变，是「O(n^{2/3}) → O(n^{1/3})」的硬核叙事。但需用讲究的波前序（如对角双源 BFS 排序）；朴素单源 BFS 只降 786→687（~12%），因 stdBrick 单元耦合对角节点、同一 BFS 壳层内仍留长链，已验证该直觉不够用；
   - **B. 块 Jacobi ILU0**：沿长轴分块、各块独立分解 + 并行三角求解，牺牲少量收敛换彻底并行，是 GPU 上最常用的 ILU 提速手段；
   - **C. 稀疏近似逆（SPAI）**：把预条件做成一个稀疏矩阵，apply 退化成一次 SpMV，彻底绕开三角求解的串行依赖；
   - **D. 其它**：SPD 场景 ILU0 退化为 IC0 省一半；`csrsv2` 在新驱动/CUDA 版本下重验。

## 七、复现方式

```bash
cd ~/OpenSees
BIN=build-cuda/OpenSees

# S1 静力（217,683 DOF）
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-ilu0-cg 18 18 200
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-ilu0-bicgstab 18 18 200

# S2 框架静力（43,350 DOF）
$BIN bench/models/s2_frame3d_static.tcl cusp-ilu0-cg 16 16 24
$BIN bench/models/s2_frame3d_static.tcl cusp-ilu0-bicgstab 16 16 24

# S3 瞬态 200 步（14,883 DOF）
$BIN bench/models/s3_brick_transient.tcl cusp-ilu0-cg 10 10 40 200
$BIN bench/models/s3_brick_transient.tcl cusp-ilu0-bicgstab 10 10 40 200
```

> 串行 SuperLU 与并行 Mumps（SCOTCH 排序）基线数据直接沿用 `phaseA_benchmark_log.md`，构建与排序器说明见其附录 A。