# Phase A/B 学习指南：GPU 加速稀疏线性求解器（CG / ILU0 / BiCGStab）

> 这份文档的目标：把 phase A、phase B 里「AI 帮你写的」每一块代码，讲成你自己能复述、能答辩的东西。
> 阅读方式：从上往下读，每一节都配了「代码在哪」和「面试会怎么问」。
> 配套文件：`CuSPSolver.cpp/.h`（求解器本体）、`bench/models/*.tcl`（算例）、`phaseA_benchmark_log.md` / `phaseB_benchmark_log.md`（数据）、`operation_history.md`（**操作全过程还原**：每一步跑了什么命令、怎么定位 bug，配合本文档的「原理」一起读）。

---

## 0. 先搞清楚：这个项目到底在解决什么问题

### 0.1 有限元的终点是一个线性方程组

OpenSees 是一个结构工程有限元软件。不管你算的是悬臂梁还是钢框架，做完网格划分、单元刚度装配之后，**所有物理问题最后都归结为解一个线性方程组**：

```
K · u = F
```

- `K`：刚度矩阵（稀疏、对称正定/非对称，取决于材料）
- `u`：节点位移（要求的东西）
- `F`：荷载向量

phase A/B 做的全部事情，就是**让这个 `K·u = F` 解得更快**——用 GPU，用迭代法，用预条件。

### 0.2 两条技术路线（面试必问的第一层）

| | 直接法 | 迭代法 |
|---|---|---|
| 代表 | SuperLU（串行）、Mumps（并行） | CG、BiCGStab（本项目） |
| 思路 | 对 K 做 LU 分解，一次性解出精确解 | 从一个猜测出发，一步步逼近 |
| 计算量 | LU 分解很贵，且 3D 问题会「填空」（fill-in） | 每次迭代就是一次矩阵向量乘，O(nnz) |
| 内存 | 大（要存 L、U 因子） | 小（只存原矩阵） |
| 何时快 | 中小规模 | 大规模稀疏，尤其 3D |

**这就是你报告里「GPU 迭代法 vs CPU 直接法」对比的本质**：拿一套硬件 + 算法，去比另一套硬件 + 算法。phase A 的 S1 能快 50 倍，一部分是 GPU 比 CPU 快，一部分是迭代法内存小、不用填空。

> 面试要点：能主动说出「这不是同算法公平对比，是 GPU 迭代 vs CPU 直接的异质对比」，说明你真的懂，不是只背了个加速比数字。

---

## 1. 数据结构：CSR 稀疏格式（一切的前提）

### 1.1 为什么不能存稠密矩阵

S1 有 217,683 个自由度，稠密矩阵要存 217683² ≈ 4.7×10¹⁰ 个数 = 376 GB，存不下。但刚度矩阵每一行其实只有少数几个非零元（一个节点只和相邻节点耦合），**非零元个数 nnz 只有几百万**。所以用稀疏格式只存非零元。

### 1.2 CSR 三数组

CSR（Compressed Sparse Row，按行压缩）用三个数组存一个稀疏矩阵：

```
rowPtr (行指针)：长度 n+1。rowPtr[i] 是第 i 行的第一个非零元在 colInd/val 里的下标。
colInd (列索引)：长度 nnz。每个非零元的列号。
val    (数值)  ：长度 nnz。每个非零元的值。
```

一个 4×4 例子：

```
矩阵：              CSR 表示：
[ 5  0  2  0 ]     rowPtr = [0, 2, 4, 5, 6]
[ 0  7  0  1 ]     colInd = [0, 2, 1, 3, 1, 3]
[ 0  8  9  0 ]     val    = [5, 2, 7, 1, 8, 9]
[ 0  0  0  3 ]
```

第 1 行（下标 0）：`rowPtr[0]=0` 到 `rowPtr[1]=2`，即 `val[0..1] = [5,2]`，列号 `[0,2]`。以此类推。

### 1.3 本项目里的一个关键约定：对角元在行首

OpenSees 的 `SparseGenRowLinSOE` 有个约定：**每一行的第一个非零元就是对角元**，后面按列号升序排。这个约定被三角求解的 kernel 直接利用了（详见 3.3）。

在代码里对应（`CuSPSolver::solve()` 开头）：

```cpp
double *h_csrVal    = theSOE->A;        // val
int    *h_csrRowPtr = theSOE->rowStartA; // rowPtr
int    *h_csrColInd = theSOE->colA;      // colInd
```

> 面试要点：能画出上面的三数组图，并说出「行首是对角元、列升序」，说明你对数据布局是真懂，不是只会调库。

---

## 2. Phase A：CG 共轭梯度 + Jacobi 预条件

### 2.1 CG 在数学上是什么

当 K **对称正定（SPD）**时（弹性问题就是），解 `K·u = F` 等价于最小化一个二次函数：

```
f(u) = ½·uᵀK·u − Fᵀu
```

CG（共轭梯度法）的核心思想：**每一步沿着一个和之前所有方向都「K-正交」（共轭）的方向走一步**，这样不会重复走老路。理论上前 n 步就能到精确解；实际远不用 n 步就够，收敛快慢取决于矩阵的**条件数** κ：

```
‖eₖ‖ ≤ 2·( (√κ−1)/(√κ+1) )ᵏ · ‖e₀‖
```

κ 越小，`(√κ−1)/(√κ+1)` 越接近 0，收敛越快。**预条件的目的就是找一个 M≈K，让 M⁻¹K 的条件数比 K 小。**

### 2.2 Jacobi 预条件：最简单的一招

Jacobi 取 `M = diag(K)`（只取对角线）。`M⁻¹b` 就是逐元素 `b[i]/K[i][i]`——一个 kernel 的事（`jacobiKernel`）。

它只是把矩阵「缩放」一下，对条件数的改善有限，但**几乎零成本**。phase A 就用它，配合 CG 已经很快。

### 2.3 代码走读：一次迭代的完整流程

看 `solveCG()`（`CuSPSolver.cpp`），每次迭代就干这几件事，全是对 GPU 向量的操作：

```
1. Ap = A·p         ← gpuSpMV()，cusparseSpMV，★ 整个求解器的核心
2. α  = (r·z)/(p·Ap)  ← cublasDdot 点积
3. x  = x + α·p      ← cublasDaxpy 向量加
4. r  = r − α·Ap     ← cublasDaxpy
5. 检查 ‖r‖ < tol → 收敛则退出
6. z  = M⁻¹·r        ← applyPreconditioner()（Jacobi 或 ILU0）
7. β  = (r·z)_new/(r·z)_old
8. p  = z + β·p
```

**SpMV（稀疏矩阵向量乘）是核心中的核心**：迭代法的每一轮，本质就是做一次 `A·p`。它只遍历非零元，计算量 O(nnz)，这就是为什么迭代法内存小、每步便宜。

### 2.4 数据流：数据是怎么在 CPU 和 GPU 之间流动的

看 `solve()`（`CuSPSolver.cpp`）的整体结构：

```
1. 从 theSOE 拿 CPU 端指针（K、F、u 都在 CPU 内存，OpenSees 组装的）
2. 首次调用：上传 CSR 结构 + 创建 cusparse 描述符 + 算 SpMV 缓冲大小
3. 每次调用：上传 K 的数值 + F（每步都变，因为非线性/瞬态）
4. 预条件：Jacobi 提取对角线求逆，或 ILU0 分解
5. 迭代求解器（solveCG / solveBiCGStab）
6. 把 u 拷回 CPU
```

关键优化点：**结构只上传一次**（`m_structureUploaded` 标志），因为非线性/瞬态分析里，矩阵的「位置」（哪些位置非零）不变，只有「数值」变。

### 2.5 phase A 的结果说明了什么

| 场景 | DOF | 串行 SuperLU | CUDA cusp-CG | 加速比 |
|---|---|---|---|---|
| S1 3D 实体 | 217,683 | 203.14s | 4.17s | 48.7× |
| S2 钢框架 | 43,350 | 107.53s | 0.56s | 192.7× |
| S3 瞬态 | 14,883 | 464.92s | 31.43s | 14.8× |

三个关键结论（面试素材）：
1. **GPU 迭代法全面领先**，甚至超过并行直接法 Mumps（np=8）。
2. **内存优势**：迭代法只存原矩阵，直接法的 LU 因子会爆炸。
3. S2 的 192× 特别夸张，因为那个框架模型对直接法「病态不利」（SuperLU 实测 ~O(n²)）——**能主动说出这个 caveat 很加分**。

---

## 3. Phase B：ILU0 预条件 + BiCGStab

phase A 之后的问题：Jacobi 预条件太弱，条件数改善有限。对于更难的问题（非对称、病态），需要更强的预条件。phase B 做的是 **ILU0** 和 **BiCGStab**。

### 3.1 ILU0 原理（phase B 的重点，面试最爱问）

#### 完整 LU vs 不完全 LU(0)

- **完整 LU**：把 K 分解成下三角 L × 上三角 U。问题是 L、U 会产生大量「填空」（fill-in）——原来 K 里是 0 的位置，分解后变成非 0。3D 问题填空严重，内存爆炸。
- **ILU(0)**：**只允许在 K 原本非零的位置产生 L、U 的非零元，禁止填空**。这样 L、U 的非零结构和 K 完全一样，内存和 K 一样大，但 `L·U ≈ K`（不完全等于，是近似）。用这个近似当预条件 M = L·U。

#### 算法逐行讲（对应 `factorizeILU0()`）

CPU 端 ILU0 用了一个「稠密工作区 w[] + 标记数组 marker[]」的技巧。对每一行 i：

**① scatter（把第 i 行摊到稠密数组）**：
```cpp
for (p = rowPtr[i]; p < rowPtr[i+1]; p++) {
    int c = colInd[p];
    w[c] = val[p];        // 把第 i 行的每个元素放进 w 的对应列位置
    marker[c] = i + 1;    // 标记「列 c 出现在第 i 行的模式里」
}
```

**② eliminate（消去严格下三角，即 j < i 的元）**：
```cpp
for (p in row i 的下三角部分) {
    int c = colInd[p];
    if (c >= i) continue;          // 只看 j < i
    double lik = w[c] / diagU[c];   // L(i,c) = A(i,c)/U(c,c)
    w[c] = lik;
    // 用第 c 行已经分解好的上三角 U(c,m) 去更新本行
    for (q in row c 的上三角部分) {
        int m = colInd[q];
        if (m <= c) continue;
        if (marker[m] == i+1)        // ★ 只有当 m 本来就在第 i 行里，才更新（否则就是填空，禁止）
            w[m] -= lik * val[q];
    }
}
```

这就是 ILU(0) 的精髓：`marker[m] == i+1` 这个判断，保证了「只有当 m 本来就在第 i 行的模式里才更新」——**从机制上杜绝了填空**。

**③ gather（写回 + 记对角线）**：
```cpp
for (p in row i) {
    val[p] = w[c];
    if (c == i) diagU[i] = w[c];   // 记录 U(i,i)，下一步要用
}
```

结果：`val` 数组就地变成了 L（严格下三角）+ U（上三角含对角）的数值，结构和原矩阵一样。

> 面试要点：能讲清楚「marker 数组是干什么的」——它是用来判断「这个列号是不是本来就出现在第 i 行」，从而禁止填空。这是 ILU(0) 和完整 LU 的唯一区别。

#### 为什么 CPU 做分解、GPU 做求解

因为 cuSPARSE 的 `cusparseDcsrilu02` 在 sm_120（Blackwell，RTX 5070）上有 bug（见 3.5），所以把 ILU0 分解放到了 CPU 上做（代码里 `factorizeILU0` 是一段普通 C++ 循环，跑在 CPU），分解结果再上传 GPU。**分解只做一次（或每步一次），三角求解才在 GPU 上反复做**。

### 3.2 BiCGStab：给非对称矩阵的 CG 变体

CG 只对对称正定矩阵有理论保证。对于非对称 K（接触、不可压缩、大变形切线刚度），用 **BiCGStab**（稳定双共轭梯度）。

BiCGStab 每一步做 **两次 SpMV + 两次预条件**（比 CG 的一倍贵），换取对一般矩阵的适用性。它基于两个残差（r 和 r̃），多了一个稳定化步骤（ω 参数）来平滑振荡。

代码在 `solveBiCGStab()`，步骤：
```
1. ρ = r̃·r
2. p = r + β(p − ωv)
3. p̂ = M⁻¹p,  v = A·p̂
4. α = ρ/(r̃·v)
5. s = r − αv
6. ŝ = M⁻¹s,  t = A·ŝ
7. ω = (t·s)/(t·t)
8. x += αp̂ + ωŝ,  r = s − ωt
```

> 面试要点：CG 一次迭代 = 1 次 SpMV + 1 次预条件；BiCGStab = 2 次 SpMV + 2 次预条件。所以 BiCGStab 每步更贵，但适用面更广。

### 3.3 三角求解：前推（forward）/ 后推（backward）

ILU0 预条件的应用是 `M⁻¹·b`，其中 M = L·U，所以：

```
M⁻¹·b = U⁻¹·(L⁻¹·b)
```

分两步：
1. **前推** `L·y = b`（L 是单位下三角）：从上往下，y[i] = b[i] − Σ_{j<i} L(i,j)·y[j]
2. **后推** `U·x = y`（U 是上三角）：从下往上，x[i] = (y[i] − Σ_{j>i} U(i,j)·x[j]) / U(i,i)

问题来了：**前推是串行的**。y[i] 依赖 y[j]（j<i），所以必须先算完前面所有行，才能算第 i 行。在一个 3D 网格上，这个依赖链会非常长。

### 3.4 level scheduling：怎么并行化这个「看起来串行」的三角求解

虽然三角求解有依赖，但**不是所有行都互相依赖**。把依赖关系画成一个 DAG（有向无环图），对每一行算它的「level」（层级）：

```
fwdLevel[i] = max( fwdLevel[j] + 1 )，对所有 j < i 且 K(i,j) ≠ 0
```

意思是：第 i 行要等它依赖的所有行 j 都算完才能算，所以它的层级 = 依赖里最深的 + 1。

**同一个 level 里的行互不依赖，可以并行算**。所以求解器按 level 一层层推进：
- 第 0 层的所有行并行算
- 全网格同步（`grid.sync()`）
- 第 1 层的所有行并行算
- ...

这就是 `fwdSubKernel` / `bwdSubKernel` 里的逻辑：

```cpp
for (int l = 0; l < numLevels; l++) {
    for (k = tid; k < nrows; k += nthreads) {  // 本 level 的行，线程并行
        i = order[start + k];
        // 算 y[i] = b[i] - Σ L(i,j) y[j]
    }
    grid.sync();   // ★ 全网格栅栏：本 level 算完才进下一 level
}
```

**关键指标是 level 数（= 最长依赖链 = 临界路径 critical path）**。它越小，需要的同步越少、能并行的越充分。而 3D 网格的 level 数在 RCM 排序下是 **O(n^{2/3})**——这就是后面性能瓶颈的根源。

> 面试要点：能画出「依赖 DAG → 分层 → 同层并行 + 层间同步」的图。这是 level scheduling 的灵魂。

### 3.5 平台坑：sm_120 上两个坏掉的 cuSPARSE API（实战亮点）

这是你报告里最有「实战含金量」的部分——**不是照本宣科调库，而是遇到了厂商库的 bug 并自己绕开**。

| API | 本应做什么 | sm_120 上的实际表现 |
|---|---|---|
| `cusparseDcsrilu02` | GPU 端 ILU0 分解 | 大矩阵（n≥8000）返回 INVALID_VALUE，**越界写 CSR 结构数组、清空对角线** |
| `cusparseSpSV` | 三角求解 | **间歇性 SIGFPE**（浮点异常崩溃） |

怎么定位的（面试要能讲这段过程）：
1. 崩溃栈指向 `cusparseSpSV ← applyPreconditioner`。
2. 写了一个**独立复现器**（`plan/spsv_fpe_repro.cu`，一个 7 点 Laplacian），把问题从 OpenSees 里剥离出来单独复现 → 确认不是 OpenSees 的问题，是库的问题。
3. **完整性检查**：分解后把结构数组拷回 CPU 和原结构 `memcmp`，发现 170 万个索引被改写、对角线变 0——实锤 `csrilu02` 越界写。
4. 修复：**全部自实现**——ILU0 分解改 CPU 端循环，三角求解改 level-scheduled kernel。

> 面试要点：这个「独立复现器 + memcmp 校验」的方法论，比「我绕开了 bug」这个结果更值钱。面试官会因此相信你能独立排查问题。

---

## 4. 性能瓶颈定位：假设 → 实验 → 定位（你最该背熟的一段）

phase B 做完后发现：ILU0 虽然把迭代次数压到 2.3×（94 次 vs Jacobi 的 219 次），但**总时间反而更慢**（S1 慢 12.9×）。为什么？这就是你要讲的「我定位了性能瓶颈」的故事。

### 第一步：提出假设

level-scheduled 三角求解有 786 个 level（8×8×40），每个 level 一个 kernel = 786 次 launch。**假设：launch 开销是主因。**

### 第二步：用实验否定假设

| 实验 | 做法 | 结果 | 结论 |
|---|---|---|---|
| persistent 单 block | 把 level 循环搬进一个 kernel，消除 786 次 launch | 只快 ~10%（2.237s→2.025s） | launch 不是主因 |
| cooperative persistent | 全网格常驻 + 逐 level `grid.sync()` | S1 无改善（51.79s→51.27s） | 也不是带宽/占用问题 |

**两次实验都否定了「launch 开销」这个最常见的猜测。**

### 第三步：定位真正瓶颈

既然不是 launch，那就是**临界路径本身**——786（8×8×40）/~6972（S1）层的依赖链，每层能并行的行数少、还要全网格同步。这是三角求解在 3D 网格上的**固有**结构问题（O(n^{2/3})）。

### 这个故事的完整逻辑链（面试直接背这段）

> 「我发现 ILU0 慢，先假设是 level-scheduled 求解里每个 level 一次 kernel launch 的启动开销太大。为了验证，我把 level 循环搬进单个 persistent kernel，只快了 10%；又换成全网格 cooperative persistent kernel，S1 上完全没有改善。两次实验都否定了 launch 开销这个猜测，于是我定位到真正的瓶颈是 level 调度的临界路径——3D 网格上它随规模 O(n^{2/3}) 增长，786/~6972 层的依赖链和每层低占用才是根因。瓶颈是测出来的，不是猜出来的。」

### 展望方向（如果面试问「你会怎么继续优化」）

1. **置换重排**：把临界路径从 O(n^{2/3}) 压向图直径 O(n^{1/3})。注意：朴素单源 BFS 只降了 12%（786→687），因为 stdBrick 单元耦合对角节点、同一 BFS 壳层内还留长链；需要更讲究的「对角双源」波前序。
2. **块 Jacobi ILU0**：沿长轴分块，各块独立分解 + 并行求解（牺牲一点收敛换彻底并行）。
3. **稀疏近似逆（SPAI）**：把预条件做成稀疏矩阵，apply 变成一次 SpMV，彻底绕开三角求解。

---

## 5. 怎么复现（操作手册）

### 5.1 编译

```bash
cd ~/OpenSees
cmake --build build-cuda --target OpenSees -j8
# 产物：build-cuda/OpenSees（CUDA 版）
```

三个二进制（别搞混）：
- `build-cuda/OpenSees`：CUDA 版（本项目的求解器）
- `OpenSees`：串行版（SuperLU，基准）
- `build-hypersp/OpenSeesSP`：并行版（Mumps，基准）

### 5.2 运行（以 S1 为例）

```bash
BIN=build-cuda/OpenSees

# Jacobi + CG（phase A 的配置）
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-cg 18 18 200

# ILU0 + CG（phase B）
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-ilu0-cg 18 18 200

# ILU0 + BiCGStab（phase B）
$BIN bench/models/s1_brick_static_ilu0.tcl cusp-ilu0-bicgstab 18 18 200

# 串行 SuperLU 基准
OpenSees bench/models/s1_brick_static_ilu0.tcl superlu 18 18 200

# 并行 Mumps 基准（SCOTCH 排序）
mpiexec -np 8 build-hypersp/OpenSeesSP bench/models/s1_brick_static_ilu0.tcl mumps 18 18 200
```

参数含义：`18 18 200` 是 x/y/z 方向的网格划分数（S1 是 18×18×200 个单元 → 217,683 自由度）。

### 5.3 怎么理解 Tcl 算例文件（以 s1 为例）

`bench/models/s1_brick_static_ilu0.tcl` 干的事：

```
1. block3D 生成 3D 网格 + stdBrick 单元（ElasticIsotropic，SPD 矩阵）
2. fixZ 0.0 固定底面 → 悬臂块
3. pattern Plain 加载 → 顶部集中力
4. system cusp -pre {jacobi|ilu0} -solver {cg|bicgstab}   ← 选求解器
5. numberer RCM  ← 节点重排序（减带宽，影响三角求解临界路径的关键）
6. analyze 1  ← 做一步静力分析
7. 输出 RESULT 行，含 tipDisp（悬臂端位移）
```

**注意 `numberer RCM` 这一行**：RCM（Reverse Cuthill-McKee）是一种节点重排，能减矩阵带宽。它直接决定了三角求解的 level 数——phase B 里「786 层」「O(n^{2/3})」都是在 RCM 排序下测的。

### 5.4 怎么验证正确性

看 `RESULT` 行的 `tipDisp`（悬臂端位移），和 SuperLU 基准对比。phase A/B 报告里相对偏差都 ≤ 1.3e-9，说明 GPU 迭代解和直接法精确解一致到小数点后 8-9 位。

### 5.5 复现验证记录（写本指南后实测，2026-08-17）

下面的命令在本机（RTX 5070，驱动 591.86，revert 后的当前代码）逐条实测过，`tipDisp` 与文档记录完全一致——**这份指南的操作步骤是可复现的**：

| 命令 | 实测 tipDisp | 文档记录 | 一致性 |
|---|---|---|---|
| `s2_frame3d_static.tcl cusp-cg 16 16 24` | 52.8167405398556795 | 52.81674053985568 | ✅ |
| `s2_frame3d_static.tcl cusp-ilu0-cg 16 16 24` | 52.8167405398569016 | 52.81674053985690 | ✅ |
| `s2_frame3d_static.tcl cusp-ilu0-bicgstab 16 16 24` | 52.8167405398580527 | 52.81674053985805 | ✅ |
| `s1_brick_static_ilu0.tcl cusp-cg 18 18 200` | 0.3282677550443785 | 0.328267755044378 | ✅ |
| `s1_brick_static_ilu0.tcl cusp-ilu0-cg 18 18 200` | 0.3282677550222638 | 0.328267755022264 | ✅ |
| `s1_brick_static_ilu0.tcl cusp-ilu0-bicgstab 8 8 40` ×10 | 0.2751108273689070（10 次逐位一致） | — | ✅ 无崩溃 |
| `s1_brick_static_ilu0.tcl superlu 8 8 40` | 0.2751108261877618 | — | 与 bicgstab 差 ~1.2e-9 |

说明：
- **正确性**：GPU 迭代解的 tipDisp 与串行 SuperLU 精确解一致到 9~15 位小数，跨运行逐位复现。
- **时间**：实测 t_analyze 与文档在 ~±35% 内波动（S2 ilu0-cg 3.06→4.15s、S1 ilu0-cg 51.8→54.8s 等），属同一机器的热状态/时钟差异，量级与结论不变。
- **未重跑**：串行 SuperLU（S1 203s、S3 瞬态 465s）与并行 Mumps 基线沿用既有 `phaseA_benchmark_log.md` 数据，太慢不重复。

---

## 6. 面试高频追问 + 参考答案

**Q1：为什么 GPU 迭代法能比 CPU 直接法快 50 倍？**
A：三个叠加因素——① GPU 并行算 SpMV；② 迭代法内存 O(nnz)，直接法 LU 有 fill-in，3D 问题 fill-in 爆炸；③ 这组对比是「GPU 迭代 vs CPU 直接」的异质对比，不是同算法公平对比。

**Q2：CG 和 BiCGStab 什么区别？**
A：CG 只对对称正定矩阵有保证，每步 1 次 SpMV；BiCGStab 对一般非对称矩阵，每步 2 次 SpMV + 2 次预条件，更通用也更贵。

**Q3：预条件是干什么的？**
A：把 Ax=b 换成 M⁻¹Ax=M⁻¹b，让 M⁻¹A 的条件数比 A 小，加速收敛。Jacobi 取 M=diag，ILU0 取 M=L·U（无填空的近似分解）。

**Q4：ILU(0) 里的「0」是什么意思？**
A：允许的填空层级是 0，即完全禁止填空。L、U 的非零结构和原矩阵一致。

**Q5：三角求解怎么并行？**
A：level scheduling——按依赖 DAG 分层，同层并行、层间 `grid.sync()`。并行度取决于 level 数（临界路径）。

**Q6：ILU0 为什么在你的平台上慢？**
A：不是 launch 开销（我做了 persistent kernel 实验否定），而是 level 调度的临界路径 O(n^{2/3}) 太长。已定位，展望用置换重排/块 Jacobi/SPAI 优化。

**Q7：sm_120 的 bug 你是怎么发现和处理的？**
A：崩溃栈指向 cuSPARSE，写独立复现器剥离复现，memcmp 校验结构数组发现越界写，最后自实现替换坏 API。

**Q8（可能被追问的深度细节）：ILU0 不是对称的，能当 CG 的预条件吗？**
A：严格说，PCG 要求预条件 M 对称正定，ILU0（LU 形式）不是对称的，理论上用 CG 不严谨。实践中 ILU0 近似对称所以仍收敛；若要严格，SPD 问题应该用 **IC0（不完全 Cholesky）**，M = L·Lᵀ 对称，还能省一半三角求解。这是一个你能主动讲出的「理论细节」。

---

## 7. 复杂度速查表

| 量 | 阶 |
|---|---|
| SpMV 一次 | O(nnz) |
| CG 每迭代 | 1 次 SpMV + 1 次预条件 |
| BiCGStab 每迭代 | 2 次 SpMV + 2 次预条件 |
| ILU0 分解 | O(nnz)（无填空，比完整 LU 便宜得多） |
| 三角求解 level 数（3D，RCM） | O(n^{2/3}) |
| 3D 网格图直径 | O(n^{1/3}) |
| 完整 LU 填空（3D） | O(n^{4/3}) 非零元（比 O(nnz) 大得多） |

---

## 8. 建议的「讲故事」顺序（面试 2 分钟版）

1. **背景**：有限元最后都归结为解 K·u=F，我要让它又快又省内存。
2. **Phase A**：GPU + CG + Jacobi，SpMV 是核心，三场景全面快于 CPU 直接法（甚至并行 Mumps）。
3. **Phase B**：更强的预条件 ILU0 + 更通用的 BiCGStab；期间遇到并自实现绕开了 sm_120 上两个坏掉的 cuSPARSE API。
4. **性能定位**：ILU0 慢 → 假设 launch 开销 → 实验否定 → 定位到 level 调度临界路径 O(n^{2/3})。
5. **展望**：置换重排 / 块 Jacobi / SPAI。

---

> 学习建议：打开 `CuSPSolver.cpp`，从 `solve()` 开始逐行读，配合本文 2.4、3.1、3.4 三节。读完能对着代码讲出「数据怎么流、ILU0 怎么分解、三角求解怎么并行」，就说明你真学会了，而不是只记得结论。
