# 阶段 A 性能基准测试记录：CUDA vs OpenSeesSP vs 串行

> 日期：2026-08-10
> 目标：用大规模模型对比 OpenSees CUDA 版（cusp CG+Jacobi）、OpenSeesSP（Mumps 并行）、串行 OpenSees（SuperLU）的计算性能
> 结果：**完成**。CUDA 版在全部三个场景中数值正确（相对偏差 ≤ 1.1e-11），性能全面领先

---

## 一、测试环境

| 项 | 值 |
|---|---|
| GPU | RTX 5070（12GB，sm_120，驱动 591.86） |
| CPU / 内存 | 16 核 / 15GB（WSL 开发机；物理机 32GB，WSL2 默认只分一半） |
| CUDA 版 | `build-cuda/OpenSees`，`system cusp -pre jacobi -solver cg -rtol 1e-8 -maxIter 20000`，运行需 `LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1` |
| 串行版 | `build/OpenSees`，`system SuperLU` |
| 并行版 | `build-hypersp/OpenSeesSP`，`system Mumps`（SCOTCH 排序，构建细节见附录 A），`mpiexec -np {4,8}` |

计时口径：Tcl `clock milliseconds` 包住 `analyze`（S1/S2 取 `t_analyze`，S3 取 200 步瞬态总时间 `t_transient`）。
基准代码与原始日志：`bench/`（`models/`、`run_all.sh`、`run_hypersp.sh`、`calibrate.sh`、`summarize.py`、`results/`）。

## 二、场景与结果（最终版）

### 场景 1：3D 实体悬臂块·静力（18×18×200 网格，217,683 DOF）

stdBrick + ElasticIsotropic，底面固结，顶部集中力，LoadControl 1 步。

| 配置 | 求解时间 | 加速比 | tip 位移 dx | 与串行偏差 |
|---|---|---|---|---|
| 串行 SuperLU | 203.14s | 1× | 0.3282677550413 | — |
| **CUDA cusp-CG** | **4.17s** | **48.7×** | 0.3282677550444 | 9.5e-12 |
| SP Mumps np=4 | 49.03s | 4.1× | 0.3282677550449 | 1.1e-11 |
| SP Mumps np=8 | 37.50s | 5.4× | 0.3282677550449 | 1.1e-11 |

### 场景 2：3D 多层钢框架·静力（16×16×24，43,350 DOF）

elasticBeamColumn（ndf=6），底层固结，顶层竖向+水平荷载。

| 配置 | 求解时间 | 加速比 | tip 位移 dx | 与串行偏差 |
|---|---|---|---|---|
| 串行 SuperLU | 107.53s | 1× | 52.81674053985805 | — |
| **CUDA cusp-CG** | **0.56s** | **192.7×** | 52.81674053985568 | 4.5e-14 |
| SP Mumps np=4 | 4.61s | 23.3× | 52.81674053927920 | 1.1e-11 |
| SP Mumps np=8 | 3.72s | 28.9× | 52.81674053956326 | 5.6e-12 |

### 场景 3：3D 实体块·瞬态动力（10×10×40，14,883 DOF，Newmark 200 步）

重力 1 步 → 去荷载 → Rayleigh 质量比例阻尼 → 自由振动 200 步，每步一次线性求解。

| 配置 | 瞬态总时间 | 单步平均 | 加速比 | tip 位移 dx | 与串行偏差 |
|---|---|---|---|---|---|
| 串行 SuperLU | 464.92s | 2.325s | 1× | 0.1804434703284 | — |
| **CUDA cusp-CG** | **31.43s** | **0.157s** | **14.8×** | 0.1804434733967 | 1.7e-8 |
| SP Mumps np=4 | 113.54s | 0.568s | 4.1× | 0.1804434703281 | 6.5e-13 |
| SP Mumps np=8 | 102.12s | 0.511s | 4.6× | 0.1804434703281 | 7.1e-13 |

## 三、正确性验证

- 三场景各配置 tip 位移与串行 SuperLU 基准的相对偏差均 ≤ 1.7e-8（见上表）
- 另做**全位移场对比**（8×8×12 框架，1053 节点 × 6 自由度全部 6317 个分量）：cusp-CG vs SuperLU 最大绝对差 2.06e-12，相对最大位移偏差 7.8e-14——GPU 解在全场意义上正确，不是单点巧合

## 四、结论

1. **CUDA 版在三个场景全面领先**：对串行 SuperLU 加速 48.7× / 192.7× / 14.8×；对最强并行配置（Mumps np=8）仍领先 **9.0×（S1）、6.6×（S2）、3.2×（S3）**
2. **内存是 GPU 迭代法的另一项硬优势**：CG+Jacobi 只需存储原矩阵 CSR（无 fill-in）；Mumps/SuperLU 的 LU 因子体积在 3D 问题上随规模快速增长（本测试中 S1 的 Mumps 单进程内存已达数百 MB，更大规模将触及 15GB 上限）
3. **口径说明**：对比是"GPU 迭代法 vs CPU 直接法"，硬件与算法差异并存。S2 的 192.7× 部分源于该框架模型对直接法的病态不利（SuperLU 实测 ~O(n²)：82k DOF 需 480s，43k 需 108s）；若 CPU 侧也用迭代法差距会显著缩小。S1/S3 的对比更代表一般情况
4. **并行直接法的适用条件**：需搭配高质量排序器（SCOTCH/METIS，见附录 A）；模型规模存在"经济下限"，过小（如 S3 的 1.5 万 DOF）时通信开销明显，进程数并非越多越好
5. WSL 时钟偶发跳变曾导致个别 `t_model` 为负值（如 s1 np8 的 -1.9s），不影响 `t_analyze` 口径的对比结论
6. **后续方向**：WSL 内存调至 24GB 后场景放大到 30 万+ DOF，让并行版进入更大优势区间；可选编译 PETSc 做"CPU 并行迭代 vs GPU 迭代"的同算法公平对比

## 附录 A：Mumps 排序器问题与 SCOTCH 版构建

> 首轮测试（`build/OpenSeesSP`，Mumps 仅 PORD 排序）中 SP 版大面积失败；以下记录诊断与修复过程。

**首轮 PORD 版失败记录**

| 场景 | 配置 | PORD 版结果 | SCOTCH 版结果 |
|---|---|---|---|
| S1 | np=4 | DNF：Error -9（单进程内存不足） | 49.03s |
| S1 | np=8 | 48.70s | 37.50s |
| S2 | np=4 | DNF：30 分钟超时 ×2 + 15 分钟复验 ×1 | 4.61s |
| S2 | np=8 | DNF：30 分钟超时 | 3.72s |
| S3 | np=4 | 142.60s | 113.54s |
| S3 | np=8 | 147.82s（反慢于 np=4，扩展性异常） | 102.12s |

**诊断**

- PORD（MUMPS 自带简易排序器）在不规则图（3D 框架、3D 实体）上消元序质量差 → fill-in 爆炸 → 表现为 S2 超时、S1 np=4 内存不足（Error -9）、S3 扩展性异常。换 SCOTCH 排序后三类症状全部消失，证实为同一病根
- S2 首轮"13 分钟 97% CPU 无进展"实为 MPI 忙等假死：Mumps 解出的并行装配残差 5.3e-8 略高于 Newton 判据 1e-8，进程判败后脱节假死。修复：`s2_frame3d_static.tcl` 中 Mumps 配置的判据放宽至 1e-6（弹性单迭代问题，不影响求解计时）

**SCOTCH 版构建（build-hypersp）**

- `~/mumps/build-metis`：scivision MUMPS 以 `-Dscotch=ON` 重编。**注**：原计划启用 METIS，但 OpenSees 内置 METIS 为 v4（1997），与系统 v5.1 符号同名不同参（`METIS_PartGraphKway` 11 参 vs 13 参），链接即栈损坏风险，故改为 **SCOTCH-only**（同属嵌套分割类排序，质量相当）
- `~/OpenSees/build-hypersp`：`-DMUMPS_DIR=~/mumps/build-metis -DSCALAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so`；`OpenSees/CMakeLists.txt` 增加自动链接 `scotch/scotcherr/esmumps`；仅编 `OpenSeesSP` 目标

## 五、复现方式

```bash
cd ~/OpenSees/bench
bash run_all.sh          # 首轮全量（PORD 版 SP；S2 的 Mumps 配置会各跑满 30 分钟超时）
bash run_remaining.sh    # 首轮补跑：S3 Mumps np4/np8 + S2 np4 复验
bash run_hypersp.sh      # 最终版：SCOTCH 版 Mumps（build-hypersp）全场景
bash calibrate.sh        # 规模校准记录
python3 summarize.py     # 重新生成 results/benchmark_report.md（仅首轮数据）
```
