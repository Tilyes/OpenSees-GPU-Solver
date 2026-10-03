# OpenSees-GPU-Solver

在 [OpenSees](https://github.com/OpenSees/OpenSees) 有限元框架上接入一个自研的 **cuSPARSE GPU 迭代求解器**（`CuSPSolver`），把有限元分析最终归结的那一步 —— 求解稀疏线性方程组 `K·u = F` —— 搬到 GPU 上，同时用迭代法替掉直接法，换来完成度更高的并行和更小的内存占用。

三场景实测（串行 SuperLU 为基准，RTX 5070 + WSL2 + CUDA 12.8）：

| 场景 | DOF | 串行 SuperLU | CUDA cusp-CG | 加速比 | Mumps np=8 |
|---|---|---|---|---|---|
| S1 3D 实体悬臂块·静力 | 217,683 | 203.14 s | **4.01 s** | **50.7×** | 37.50 s |
| S2 3D 多层钢框架·静力 | 43,350 | 107.53 s | **0.51 s** | **212×** | 3.72 s |
| S3 3D 实体块·瞬态 200 步 | 14,883 | 464.92 s | **34.31 s** | **13.6×** | 102.12 s |

> **EN TL;DR** — A CUDA/cuSPARSE iterative linear solver (CG & BiCGStab with Jacobi / ILU(0) preconditioning) plugged into OpenSees. Up to **212× speedup** over serial SuperLU on three 3D benchmark problems, with agreement to ~1e-9 relative error. This repository is a research fork of OpenSees and is **not** a from-scratch FE code — see §1 for upstream attribution and licensing.

---

## 一、先说清楚：这是一个"站在别人肩膀上"的项目

**OpenSees 是别人的开源项目，不是本仓库作者的成果。** 这一节把归属交代干净，避免任何误解。

| 项 | 说明 |
|---|---|
| 上游项目 | **OpenSees**（*Open System for Earthquake Engineering Simulation*） |
| 主导开发方 | 美国**太平洋地震工程研究中心（PEER）**，加州大学伯克利分校 |
| 上游仓库 | <https://github.com/OpenSees/OpenSees> |
| 上游文档 | <https://opensees.github.io/OpenSeesDocumentation> |
| 版权 | Copyright © 1999-2020 The Regents of the University of California（见 [`COPYRIGHT`](COPYRIGHT)） |
| 许可性质 | **非商业**：允许教育 / 研究 / 非营利机构用于非商业目的，允许其他机构内部使用；商业用途需另行向 UC 技术转移办公室申请授权 |

本仓库保留了上游的 **完整源码树** 与 **`COPYRIGHT` 文件**，未做任何署名上的删改。**除 §2 列出的改动之外，所有 FE 单元、材料、分析算法、Tcl 解释器、建模能力都来自上游 OpenSees，与本仓库作者无关。**

### 本仓库作者新增了什么

| 新增 | 文件 | 说明 |
|---|---|---|
| GPU 迭代求解器 | `SRC/system_of_eqn/linearSOE/sparseGEN/CuSPSolver.{h,cpp}` | 整体新写：CG / BiCGStab 迭代循环、Jacobi 与 ILU(0) 预条件、level-scheduled 三角求解、CPU 端 ILU(0) 分解 |
| Tcl 命令注册 | `SRC/tcl/commands.cpp`、`SRC/runtime/commands/analysis/solver.{cpp,hpp}` | 新增 `system cusp ...` / `system cuSPARSE ...` |
| 构建接入 | 根 `CMakeLists.txt`、`sparseGEN/CMakeLists.txt`、`runtime/commands/CMakeLists.txt` | 启用 CUDA 语言、接线 cuSPARSE / cuBLAS |
| 基准与复现 | `bench/`（4 个算例 + 15 份原始日志 + 汇总脚本） | 三场景基准，脚本化可复现 |
| 过程文档 | `plan/`（实施记录、基准记录、学习指南、复现器） | 含 sm_120 坏 API 的独立复现器 `spsv_fpe_repro.cu` |

### 引用与致谢

若你在研究中使用本工作，除了引用本仓库，**请同时引用 OpenSees 本体**：

> Mazzoni, S., McKenna, F., Scott, M. H., & Fenves, G. L. *OpenSees Command Language Manual*. Pacific Earthquake Engineering Research Center, University of California, Berkeley.

---

## 二、做了什么

### 2.1 问题

不论算悬臂梁还是钢框架，做完网格与单元装配后，物理问题最后都归结为解一个线性方程组：

```
K · u = F          K 稀疏刚度矩阵，u 节点位移，F 荷载向量
```

两条技术路线：

| | 直接法（SuperLU / Mumps） | 迭代法（本项目） |
|---|---|---|
| 思路 | LU 分解，一次解出精确解 | 从猜测出发逐步逼近 |
| 计算量 | 分解昂贵，3D 问题存在大量「填空」(fill-in) | 每步一次 SpMV，O(nnz) |
| 内存 | 大（需存 L、U 因子） | 小（只存原矩阵） |
| 何时快 | 中小规模 | 大规模稀疏，尤其 3D |

### 2.2 两阶段

| 阶段 | 内容 | 结果 |
|---|---|---|
| **Phase A** | `CuSPSolver` 跑通：CG（共轭梯度）+ Jacobi 预条件，`cusparseSpMV` 为核心 | 三场景全面快于串行 SuperLU，最大 **212×** |
| **Phase B** | 更强预条件：ILU(0) + 更通用的 BiCGStab；自实现绕开 sm_120 上两个损坏的 cuSPARSE API | 数值全部正确（≤1.3e-9），瓶颈定位到 level 调度临界路径 |

---

## 三、快速开始

### 3.1 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5070（12 GB，**sm_120**，驱动 591.86） |
| CPU / 内存 | AMD 9700X / 32 GB（WSL2 默认只分一半，实测 15 GB） |
| 系统 | Windows 11 + **WSL2**（Ubuntu） |
| CUDA | **CUDA 12.8** + cuSPARSE / cuBLAS |
| 依赖 | CMake ≥ 3.20、gcc、Tcl/Tk、Mumps（可选，用于并行基线） |

> `CMAKE_CUDA_ARCHITECTURES` 在本机设为 `"89;120"`。CUDA 12.0 不认识 sm_120，需要 12.8+ 原生支持。

### 3.2 构建

```bash
# 在 WSL 里
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build build-cuda --target OpenSees -j8

# 产物
build-cuda/OpenSees      # CUDA 版（本项目的求解器）
```

三个二进制别搞混：

| 二进制 | 含义 |
|---|---|
| `build-cuda/OpenSees` | **CUDA 版**（本项目） |
| `OpenSees` | 串行版（SuperLU 基准） |
| `build-hypersp/OpenSeesSP` | 并行版（Mumps 基准） |

构建时容易踩的坑（`plan/phaseA_implementation_log.md` 有完整记录）：

- `.cpp` 里写了 `<<<>>>` kernel 语法 → CMake 必须给该文件设 `LANGUAGE CUDA`，且要在 `set_source_files_properties` 里用 **独立参数** `TARGET_DIRECTORY`（写进 `PROPERTIES` 会被静默忽略）；
- 根 CMake 必须 `enable_language(CUDA)`，否则属性不生效、g++ 直接编译 kernel 语法报错；
- CUDA 构建要**禁用** `-static-libstdc++`，否则与 CUDA 链接布局叠加后 `std::cerr` 的 vtable 为 NULL，程序连 `puts` 都崩；
- CUDA 12 已删除全部 legacy cuSPARSE API（`cusparseDcsrmv`、`csrsv2` …），必须迁移到 generic API（`cusparseSpMV` / `cusparseSpSV`）。

### 3.3 运行

```bash
BIN=build-cuda/OpenSees

# Jacobi + CG（Phase A 配置）
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-cg 18 18 200

# ILU0 + CG / ILU0 + BiCGStab（Phase B 配置）
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-ilu0-cg       18 18 200
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-ilu0-bicgstab 18 18 200

# 串行 SuperLU 基准
OpenSees bench/models/s1_brick_static_ilu0.tcl superlu 18 18 200

# 并行 Mumps 基准
mpiexec -np 8 build-hypersp/OpenSeesSP bench/models/s1_brick_static_ilu0.tcl mumps 18 18 200
```

> **WSL 专用**：若出现 `no CUDA-capable device is detected`，是 `/lib/x86_64-linux-gnu/` 下手动装的 Linux 原生驱动库抢占了加载。用仓库根目录的包装器即可：
> ```bash
> ./opensees-cuda bench/models/s1_brick_static_ilu0.tcl cusp-cg 18 18 200
> ```

### 3.4 Tcl 用法

```tcl
system cusp -pre {jacobi|ilu0|none} -solver {cg|bicgstab} -rtol 1e-8 -maxIter 20000
```

| 参数 | 取值 | 默认 |
|---|---|---|
| `-pre` | `none`(0) / `jacobi`(1) / `ilu0`(2) | `jacobi` |
| `-solver` | `cg`(0) / `bicgstab`(1) / `gmres`(2) | `bicgstab` |
| `-rtol` / `-rTol` | 相对收敛容差 | `1e-6` |
| `-maxIter` / `-mInt` | 最大迭代次数 | `5000` |

命令名 `cusp` 与 `cuSPARSE` 均可（大小写不敏感）。

---

## 四、实现要点

### 4.1 CSR 稀疏格式与一条关键约定

OpenSees 的 `SparseGenRowLinSOE` 用 CSR 三数组存矩阵：

```
rowPtr (n+1)  第 i 行首个非零元在 colInd/val 中的下标
colInd (nnz)  每个非零元的列号
val    (nnz)  每个非零元的值
```

约定：**每行第一个非零元就是对角元，其后按列号升序**。三角求解 kernel 直接依赖这条约定。

S1 有 217,683 自由度，稠密矩阵要 217683² ≈ 4.7×10¹⁰ 个数（约 376 GB），而 CSR 只有几百万非零元 —— 这是迭代法内存优势的根源。

### 4.2 `solve()` 数据流

```
① 从 theSOE 取 CPU 端指针（K、F、u 由 OpenSees 在 CPU 组装）
② 首次调用：上传 CSR 结构 + 创建 cuSPARSE 描述符 + 算 SpMV 缓冲大小
③ 每次调用：上传 K 数值 + F（非线性/瞬态每步都变）
④ 预条件：Jacobi 提取对角线求逆；或 ILU0 分解
⑤ 迭代求解 solveCG() / solveBiCGStab()
⑥ 把 u 拷回 CPU
```

**结构只上传一次**（`m_structureUploaded` 标志）：非线性/瞬态分析里矩阵的「位置」不变，只有「数值」变。

> 这里踩过一个坑：OpenSees 先调 `setSize()`（内部把 `m_initialized` 置 true）再调 `solve()`，若用 `if (!m_initialized)` 判断首次调用，CSR 结构就永远不会上传，`d_csrRowPtr` 是垃圾指针。故另立独立标志。

### 4.3 CG + Jacobi（Phase A）

当 K 对称正定（SPD，弹性问题即是）时，解 `K·u = F` 等价于最小化 `f(u) = ½uᵀKu − Fᵀu`。CG 每步沿与之前所有方向 K-正交的方向前进：

```
r = b - A*x;  z = M⁻¹r;  p = z
for k:
    Ap = A·p                    ← gpuSpMV()，核心
    α  = (r·z)/(p·Ap)           ← cublasDdot
    x += α·p;  r -= α·Ap        ← cublasDaxpy
    if ‖r‖ < tol: break
    z  = M⁻¹r                   ← 预条件
    β  = (r·z)_new/(r·z)_old
    p  = z + β·p
```

Jacobi 取 `M = diag(K)`，`M⁻¹b` 就是逐元素 `b[i]/K[i][i]` —— 一个 kernel 搞定，几乎零成本，但只做缩放，对条件数改善有限。

### 4.4 ILU(0) + BiCGStab（Phase B）

**ILU(0)**：只允许在 K 原本非零的位置产生 L、U 的非零元，**从机制上禁止填空**，于是 L、U 与 K 结构完全一致、内存同量级，而 `L·U ≈ K` 可用作预条件。

`factorizeILU0()` 逐行做三件事：

```cpp
// ① scatter：把第 i 行摊到稠密工作区
for (p = rowPtr[i]; p < rowPtr[i+1]; p++) {
    w[colInd[p]] = val[p];
    marker[colInd[p]] = i + 1;        // 标记「列 c 出现在第 i 行」
}
// ② eliminate：消去严格下三角 j < i
lik = w[c] / diagU[c];
for (q in 第 c 行上三角) {
    if (marker[m] == i + 1)           // ★ 只在本来就在第 i 行里时才更新
        w[m] -= lik * val[q];         //   否则就是填空，跳过
}
// ③ gather：写回，val 就地变成 L + U
```

`marker` 那个判断，就是 ILU(0) 与完整 LU 的唯一区别。

**BiCGStab** 给非对称矩阵（接触、不可压缩、大变形切线刚度）用，每步 **2 次 SpMV + 2 次预条件**，比 CG 贵一倍但适用面更广。

### 4.5 level-scheduled 三角求解（并行化一个"看起来串行"的过程）

`M⁻¹b = U⁻¹(L⁻¹b)`：前推 `L·y = b` 是串行的（`y[i]` 依赖所有 `y[j], j<i`）。但并非所有行互相依赖 —— 把依赖画成 DAG 后分层：

```
fwdLevel[i] = max(fwdLevel[j] + 1)，对所有 j < i 且 K(i,j) ≠ 0
```

**同层行互不依赖，可并行；层间 `grid.sync()` 同步**：

```cpp
for (int l = 0; l < numLevels; l++) {
    for (k = tid; k < nrows; k += nthreads)   // 本层内并行
        y[order[start + k]] = b[i] - Σ L(i,j) y[j];
    grid.sync();                              // 层间栅栏
}
```

**level 数 = 最长依赖链 = 临界路径**，这是决定性能的关键量。3D 网格在 RCM 排序下它是 **O(n²ᐟ³)**。

### 4.6 sm_120 上两个损坏的 cuSPARSE API —— 自实现绕开

本平台（RTX 5070，Blackwell，sm_120）上两个官方 API 不可用：

| API | 本应做什么 | sm_120 上的实际表现 |
|---|---|---|
| `cusparseDcsrilu02` | GPU 端 ILU(0) 分解 | n≥8000 返回 `INVALID_VALUE`，**越界写 CSR 结构数组、清空对角线**，却报成功 |
| `cusparseSpSV` | GPU 端三角求解 | **间歇性 SIGFPE**（约 7/10 概率复现） |

定位方法（比结论更有价值）：

1. 崩溃栈指向 `cusparseSpSV ← applyPreconditioner`；
2. 写**独立复现器**（[`plan/spsv_fpe_repro.cu`](plan/spsv_fpe_repro.cu)，一个 7 点 Laplacian），把问题从 OpenSees 里剥离出来单独复现 → 确认不是 OpenSees 的问题；
3. **完整性校验**：分解后把 CSR 结构数组拷回 CPU 与原始结构 `memcmp` → 发现 **170 万个索引被改写、对角线变 0**，实锤 `csrilu02` 越界写；
4. 修复：**弃用这两个 API，全部自实现** —— ILU(0) 分解改 CPU 端普通 C++ 循环（`marker` + 稠密工作区），三角求解改 level-scheduled cooperative kernel。

---

## 五、性能结果

**测试口径**：Tcl `clock milliseconds` 包住 `analyze`；S1/S2 取 `t_analyze`，S3 取 200 步 Newmark 瞬态总时间。「加速比」= 相对串行 SuperLU。

### 场景 1 — 3D 实体悬臂块·静力（18×18×200，**217,683 DOF**）

stdBrick + ElasticIsotropic，底面固结，顶部集中力，LoadControl 1 步。

| 配置 | 时间 | 加速比 | tip 位移 | 与串行偏差 |
|---|---|---|---|---|
| 串行 SuperLU | 203.14 s | 1× | 0.3282677550413 | — |
| **cusp-cg (Jacobi)** | **4.007 s** | **50.7×** | 0.328267755044378 | 9.5e-12 |
| cusp-bicgstab (Jacobi) | 4.346 s | 46.7× | 0.328267755042745 | 4.5e-12 |
| cusp-ilu0-cg | 51.788 s | 3.9× | 0.328267755022264 | 1.9e-11 |
| cusp-ilu0-bicgstab | 114.998 s | 1.8× | 0.328267756335040 | 1.3e-9 |
| SP Mumps np=4 | 49.03 s | 4.1× | 0.3282677550449 | 1.1e-11 |
| SP Mumps np=8 | 37.50 s | 5.4× | 0.3282677550449 | 1.1e-11 |

### 场景 2 — 3D 多层钢框架·静力（16×16×24，**43,350 DOF**）

`elasticBeamColumn`（ndf=6），底层固结，顶层竖向+水平荷载。

| 配置 | 时间 | 加速比 | tip 位移 | 与串行偏差 |
|---|---|---|---|---|
| 串行 SuperLU | 107.53 s | 1× | 52.81674053985805 | — |
| **cusp-cg (Jacobi)** | **0.507 s** | **212×** | 52.81674053985568 | 4.5e-14 |
| cusp-bicgstab (Jacobi) | 0.778 s | 138× | 52.81674053985811 | 2.0e-15 |
| cusp-ilu0-cg | 3.06 s | 35.1× | 52.81674053985690 | 1.1e-12 |
| cusp-ilu0-bicgstab | 6.812 s | 15.8× | 52.81674053985805 | 2.7e-15 |
| SP Mumps np=8 | 3.72 s | 28.9× | 52.81674053956326 | 5.6e-12 |

### 场景 3 — 3D 实体块·瞬态动力（10×10×40，**14,883 DOF**，Newmark 200 步）

重力 1 步 → 去荷载 → Rayleigh 质量比例阻尼 → 自由振动 200 步，每步一次线性求解。

| 配置 | 瞬态总时间 | 单步平均 | 加速比 | tip 位移 | 与串行偏差 |
|---|---|---|---|---|---|
| 串行 SuperLU | 464.92 s | 2.325 s | 1× | 0.1804434703284 | — |
| **cusp-cg (Jacobi)** | **34.313 s** | **0.172 s** | **13.6×** | 0.1804434733967 | 1.7e-8 |
| cusp-ilu0-cg | 64.366 s | 0.322 s | 7.2× | 0.1804434706225 | 2.9e-10 |
| cusp-ilu0-bicgstab | 63.548 s | 0.318 s | 7.3× | 0.1804434716708 | 1.3e-9 |
| SP Mumps np=8 | 102.12 s | 0.511 s | 4.6× | 0.1804434703281 | 7.1e-13 |

### 三点必要说明（避免过度解读）

1. **这是异质对比，不是同算法公平对比。** 比的是「GPU + 迭代法」对「CPU + 直接法」，加速比里同时包含硬件差异与算法差异。真要看算法，得拿 GPU 直接法或 CPU 迭代法作对照。
2. **S2 的 212× 有场景特殊性。** 梁单元框架对直接法「病态不利」，SuperLU 实测接近 O(n²)，放大加速比。这个场景更应读作「直接法在该问题上的内存/计算劣势被暴露」，而非「GPU 普遍快 212 倍」。
3. **内存优势是硬优势。** 迭代法只存原矩阵 CSR、无 fill-in，这是与数据规模无关的结构性优势。

---

## 六、正确性验证

**三重验证，且校验整个解向量而非抽查单点：**

- **对解析解**：`test_cusp.tcl`（10 单元 2D 悬臂梁）端部挠度解析解 `P·L³/(3EI) = -11.494252873563218`，cusp-CG 得 `-11.494252873559686`（偏差 ~3e-12），cusp-BiCGStab 偏差 ~3e-11。
- **对串行 SuperLU 精确解**：三场景全部配置 tip 位移相对偏差 **≤ 1.3e-9**，远优于 Newton 迭代判据（1e-8）。
- **全场级比对**：S1 抽 **6317 个位移分量**逐点对比，**最大绝对差 2.06e-12，相对偏差 7.8e-14**。

**稳定性**：S1 全尺寸（217,683 DOF，正是 `csrilu02` 之前确定性崩溃的规模）+ 8×8×40 连续 10 次 + S3 瞬态 200 步，**无崩溃、无 `zero pivot`、无 Jacobi 回退** —— sm_120 上的间歇性 SIGFPE 已消除。

---

## 七、已知问题与限制

### 7.1 ILU(0) 在 SPD 问题上反而更慢（重要，别误用）

Phase B 最反直觉的结论：

| 场景 | cusp-cg (Jacobi) | cusp-ilu0-cg | 倍数 |
|---|---|---|---|
| S1 | 4.01 s | 51.79 s | **慢 12.9×** |
| S2 | 0.51 s | 3.06 s | 慢 6.0× |
| S3 | 34.31 s | 64.37 s | 慢 1.9× |

ILU(0) 确实把 CG 迭代次数压到了 **2.3×**（8×8×40：94 次 vs 219 次），但**每次预条件（三角求解）是 Jacobi 的 ~25×**，迭代红利被完全抵消。

**原因已定位（假设 → 实验 → 定位）**：

| 假设 | 实验 | 结果 | 结论 |
|---|---|---|---|
| 786 层 = 786 次 kernel launch，启动开销主导 | level 循环搬进单个 persistent kernel | 仅 2.237 s → 2.025 s（~10%） | **否定** |
| 单 block 带宽不足 | cooperative persistent kernel + 逐层 `grid.sync()` | S1 51.79 s → 51.27 s（无改善） | **否定** |
| → 真瓶颈 = 依赖临界路径 | — | level 数 786（8×8×40）/ ~6972（S1），O(n²ᐟ³)，每层活跃行少、占用低 | **定位** |

> 两次实验都否定了最常见的「launch 开销」猜测。**瓶颈是测出来的，不是猜出来的。**

**结论：** 本仓库三个场景均为 SPD（弹性），CG + Jacobi 本就最优，ILU(0) 的价值不在这里。ILU(0)/BiCGStab 的真正价值应在**非对称 / 病态**问题（接触、不可压缩、大变形切线刚度非对称）上体现 —— 这类场景尚未建benchmark，属已知空白。

### 7.2 对并行 Mumps 并不总是赢

S1 上 cusp-ilu0-cg（51.8 s）已输给 Mumps np=8（37.5 s）。Jacobi+CG 仍是全面领先的那个。

### 7.3 平台注意事项

- **CUDA 12.0 不够用**：不认识 sm_120，且 generic API 行为未定型。建议 **CUDA 12.8+**。
- **`cusparseCreate` 首调失败**：CUDA 12.0 已知问题 —— 若 `cusparseCreate` 是进程第一个 CUDA 调用，驱动侧上下文初始化失败（`CUDA context cannot be initialized`）。构造函数里先调一次 `cudaGetDeviceCount()` 即可规避。
- **WSL 的 libcuda 冲突**：见 §3.3。
- **子模块未包含**：`Win64/s3hark`、`OTHER/eigenAPI/eigen` 是上游的 git submodule，本仓库不包含其内容，需要时请按 `.gitmodules` 自行拉取。Windows 预编译库（`Win32/lib`、`Win64/lib`）在上游仓库中以二进制形式存在，本仓库同样保留。

---

## 八、文档索引

`plan/` 与 `bench/` 下是本项目的完整过程记录，建议按此顺序读：

| 文件 | 内容 |
|---|---|
| [`plan/phaseA_implementation_log.md`](plan/phaseA_implementation_log.md) | Phase A 实施记录：8 个 bug 的定位过程（nvcc 不生效、`std::cerr` vtable 为 NULL、CSR 结构从未上传、Blackwell `cusparseCreate` …） |
| [`plan/phaseA_benchmark_log.md`](plan/phaseA_benchmark_log.md) | Phase A 三场景基准与正确性验证 |
| [`plan/phaseB_implementation_log.md`](plan/phaseB_implementation_log.md) | Phase B 实施记录：ILU(0) 自实现与 level 调度构建 |
| [`plan/phaseB_benchmark_log.md`](plan/phaseB_benchmark_log.md) | Phase B 基准、瓶颈定位（假设→实验→定位）与结论 |
| [`plan/spsv_fpe_repro.cu`](plan/spsv_fpe_repro.cu) | **独立复现器**：把 sm_120 的 SIGFPE 从 OpenSees 里剥离出来 |
| [`plan/learning_guide.md`](plan/learning_guide.md) | 从零讲清 CG / ILU(0) / level scheduling 的原理与代码走读（含面试问答） |
| [`plan/operation_history.md`](plan/operation_history.md) | 操作全过程还原：每一步跑了什么命令、怎么定位 bug |
| [`bench/results/benchmark_report.md`](bench/results/benchmark_report.md) | 早期汇总报告（数字为中间版本，以 `phaseB_benchmark_log.md` 为准） |

`bench/results/*.log` 是 15 份原始运行日志，`bench/summarize.py` 负责汇总。

---

## 九、展望

按性价比排序：

| | 方向 | 预期 |
|---|---|---|
| **A** | **置换重排**：把三角求解临界路径从 RCM 的 O(n²ᐟ³) 压向图直径 O(n¹ᐟ³) | 不动预条件，收敛行为不变，叙事最漂亮。注意：朴素单源 BFS 只降 786→687（~12%），因 stdBrick 耦合对角节点、同壳层内仍留长链，需要「对角双源」波前序 |
| **B** | **块 Jacobi ILU(0)**：沿长轴分块，各块独立分解 + 并行求解 | 牺牲少量收敛换彻底并行，GPU 上最常用的 ILU 提速手段 |
| **C** | **稀疏近似逆 SPAI**：预条件做成稀疏矩阵，apply = 一次 SpMV | 彻底绕开三角求解的串行依赖 |
| **D** | **非 SPD 场景验证**：接触 / 不可压缩 / 大变形切线刚度 | 给 ILU(0)+BiCGStab 找到真正该用的地方（当前最大的空白） |
| **E** | SPD 场景改用 **IC(0)**（`M = L·Lᵀ`，对称） | 省一半三角求解 |

---

## 十、许可与致谢

- **上游 OpenSees**：Copyright © 1999-2020 The Regents of the University of California。许可证条款见 [`COPYRIGHT`](COPYRIGHT)，**仅限教育 / 研究 / 非营利机构的非商业用途**；商业用途请联系 UC Berkeley 技术转移办公室。
- **本仓库新增部分**（`CuSPSolver`、基准脚本、`plan/` 文档）：作者 Tilyes，随上游条款一并发布，同样限非商业研究与教育用途。
- 致谢：OpenSees 开发团队（PEER / UC Berkeley）、以及 NVIDIA cuSPARSE / cuBLAS 提供的 GPU 基础算子。
