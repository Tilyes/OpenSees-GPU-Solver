# Phase A/B 操作全过程还原（来自 Kimi Code 会话记录）

> 这份文档把「AI 帮你写的每一段代码、每一条命令」按时间顺序还原成完整的操作过程，配套 `learning_guide.md` 一起读：
> - **`learning_guide.md`** 讲**原理**（CSR 是什么、CG/ILU0/BiCGStab 数学、level scheduling 怎么并行）；
> - **本文档**讲**过程**（每一步实际跑了什么命令、出了什么错、怎么定位、怎么改）。
>
> 数据来源：`.kimi-code/sessions/wd_jumjum_6db2175d6ee8/` 下 4 个会话的完整 wire.jsonl 操作记录（313 条 Bash 命令、75 次 Edit、20 次 Write）。
>
> 时间线总览：
> | 会话 | 日期 | 干了什么 |
> |---|---|---|
> | session_6a77fdeb | 8/1–8/2 | 初始 CUDA 求解器重写 + 编译排错 |
> | session_1d8493e2 | 8/10–8/11 | Phase A 基准对比（三场景）+ CUDA 12.8 升级 |
> | session_f9486c21 | 8/11–8/12 | Phase B ILU0 + sm_120 两个坏 API 排查 |
> | session_4e82ef09 | 8/17 | 问「history 在哪」（无技术内容，略） |

---

# 会话一：初始 CUDA 求解器重写（8/1–8/2）

## 1.1 起点：你重写了求解器，AI 先读现状

你的第一句话：

> `@CuSPSolver.cpp 我重写了这个cuda加速求解器，帮我想想下一步该怎么做 @.plan.md 是设定的计划`

AI 的第一步**不是写代码，而是读现状**——先 `Read` 了 `.plan.md`、`CuSPSolver.cpp/.h`、`CMakeLists.txt`，搞清两件事：
1. 代码里已经有哪些 CUDA 调用、哪些是空壳/TODO；
2. 计划里约定要做什么（CG + Jacobi 是 phase A，ILU0 + BiCGStab 是 phase B）。

**这一步的教训**：接手一段「AI 写的代码」或「自己重写的代码」，第一件事永远是先读清楚「现在到底能不能编译、缺什么」，而不是急着往上加功能。

## 1.2 第一个障碍：CUDA 12 删了一批老 API

读代码时发现，你重写的代码里用了几个 **CUDA 11 时代、到 CUDA 12 已删除**的 API：

| 老 API | 作用 | CUDA 12 状态 | 替代 |
|---|---|---|---|
| `cusparseDcsrmv` | 稀疏矩阵向量乘 SpMV | **已删除** | `cusparseSpMV`（generic API） |
| `cusparseDcsrsv2` | 三角求解 SpSV | **已删除** | `cusparseSpSV`（generic API） |
| `cusparseDcsrilu02` | ILU0 分解 | 还在，但 deprecated | （phase B 会踩坑） |

对应操作（AI 做的几处 Edit）：
- 把 CG 里的 `Ap = A·p`、BiCGStab 里的 `v = A·p̂`、`t = A·ŝ` 三处 `cusparseDcsrmv(...)` 全部替换成自写的辅助函数 `gpuSpMV(...)`（内部调 `cusparseSpMV`），这样三处调用点统一，以后换 SpMV 实现只改一处。
- 把注释里的 `csrsv2` 全部改成 `SpSV` 的说明，标好 phase B 再实现。

> **为什么这么设计**：把 SpMV 包成 `gpuSpMV()` 一个函数，是「把易变的底层调用隔离成一个入口」——后面 phase B 要换 SpMV 或者加预条件，都不用动三个调用点。这就是面试能讲的「封装」。

## 1.3 编译排错：四个连环坑（这是会话一的主线）

### 坑 1：`_CUSP` 宏不能全局定义

OpenSees 的 `SRC/tcl/commands.cpp` 里有一块 `#ifdef _CUSP` 的**旧的 Windows-only CuSP 注册代码**，里面用了 `_stricmp` 和 DLL 加载逻辑。如果全局 `add_compile_definitions(_CUSP)`，Linux 编译就会踩进这段旧代码直接报错。

**定位**：`grep -n "_CUSP" SRC/tcl/commands.cpp` 看到那段 Windows 代码。

**修复**：`_CUSP` 只定义在需要它的目标上（`sparseGEN` 和 `runtime/commands`），不全局定义。同时把旧块里的 `_stricmp` 换成 POSIX 的 `strcasecmp`，`#include <strings.h>`。

### 坑 2：`-static-libstdc++` 与 CUDA 冲突 → 启动即段错误

编译过了，但 `OpenSees /tmp/hello.tcl` 一跑就 SIGSEGV，栈指向 `StandardStream::operator<<`。

**定位过程**（这套 gdb 手法很值得记）：
```bash
gdb -batch -ex "break std::ios_base::Init::Init()" -ex run -ex "printf \"Init ctor HIT\n\"" --args ./OpenSees hello.tcl
# 对比「CUDA 版崩溃」和「无 CUDA 版正常」两个二进制，发现：
# CUDA 版静态链了 libstdc++，std::cerr 对象没被正确初始化，第一次用就崩。
```

**根因**：CMake 里 GNU 分支写了 `CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++"`，这个静态链接标志和 CUDA 的链接布局叠加，导致可执行文件里的静态 `libstdc++` 的 `std::cerr` 全局对象初始化失败。

**修复**：CUDA 构建去掉 `-static-libstdc++`/`-static-libgcc`（CUDA 二进制本来就要依赖 CUDA 动态库，没必要静态 libstdc++）。

### 坑 3：`cusparseCreate` 作为进程第一个 CUDA 调用会崩

去掉静态链接后 hello 能跑了，但 `system cusp` 一进求解器就崩。AI 写了最小复现器 `probe2.cu` / `probe3.cu`：

```c
cusparseHandle_t h; cusparseCreate(&h);   // 整个进程的第一个 CUDA 调用
// → 在 CUDA 12.0 + Blackwell(sm_120) 上返回错误/崩
```

**定位结论**：`cusparseCreate` 如果是**进程的第一个 CUDA 调用**，驱动上下文还没初始化，就出问题。先随便调一个 `cudaFree(0)` 强制初始化运行时上下文，再 `cusparseCreate` 就正常。

**修复**：`CuSPSolver` 初始化时，`cusparseCreate` 之前先 `cudaFree(0)`（或等价地先初始化运行时上下文）。

> 这条是**平台 bug**，不是算法 bug。面试讲这段时要点出：「不是每个 CUDA 调用都安全做『第一个调用』，Blackwell 新平台 + 老 toolkit 组合尤其如此。」

### 坑 4：WSL 的 `libcuda.so.1` 是坏的

之后 `cusparseCreate` 还是报 `cusparseCreate failed`，除非加 `LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1`。

**定位**：`ldconfig -p | grep libcuda` 发现 `/lib/x86_64-linux-gnu/libcuda.so.1` 指向一个被改名禁用的实体库 `libcuda.so.580.173.02.disabled`（之前有人装过 Linux 版驱动包 `libnvidia-compute-580`，和 WSL 的 GPU 直通冲突）。

**修复**（会话一先临时绕，会话三彻底修）：
```bash
# 临时：跑 CUDA 程序时手动 preload WSL 正确的 libcuda
LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1 ./build-cuda/OpenSees test_cusp.tcl cusp
# 并写了个 wrapper 脚本 opensees-cuda 固化这个 preload
```

## 1.4 正确性首验：悬臂梁对解析解

写了个 `test_cusp.tcl`（悬臂梁），对比：
- `superlu`（直接法精确解）
- `cusp`（CUDA 迭代解）

结果 tip 位移 **-11.494252873567**，解析解 -11.494252873563 —— 对上，说明求解器链路（SpMV + CG + Jacobi + 上传下载）通了。

随后又 `sed` 出一个 `bicgstab` 变体验证 BiCGStab 分支，也通过。

## 1.5 会话一收尾

- 验证了串行 `OpenSees` 和并行 `OpenSeesSP` 两个二进制都能跑 hello（确认基准工具可用）；
- 把整个过程写进 `plan/phaseA_implementation_log.md`。

---

# 会话二：Phase A 基准对比（8/10–8/11）

你的目标：**用多个复杂大规模场景，对比 OpenSees_cuda vs OpenSeesSP（并行）vs OpenSees 串行版的性能**。

## 2.1 环境探查（先摸清硬件）

```bash
nvidia-smi      # → RTX 5070, 12GB, driver 591.86
nproc           # → 16 核
free -g         # → 15GB（WSL 只分到物理 32GB 的一半）
which mpiexec mpirun
```

**这一步决定了后面所有结论的边界**：15GB 内存、12GB 显存是硬约束，后面 Mumps np=4 内存不足、S2 超时都跟这个有关。

## 2.2 造三个基准算例（这是 phase A 的核心资产）

写了三个 `.tcl` 模型，每个都带 `solver nx ny nz` 参数化、统一输出 `RESULT scenario=... ndof=... t_analyze=... tipDisp=...`：

| 模型 | 类型 | 最终规模 | 求解器分支 |
|---|---|---|---|
| `s1_brick_static.tcl` | 3D 实体悬臂块（stdBrick + ElasticIsotropic，SPD） | 18×18×200 = 217,683 DOF | superlu / mumps / cusp-cg / cusp-bicgstab |
| `s2_frame3d_static.tcl` | 3D 多层钢框架（elasticBeamColumn，ndf=6） | 16×16×24 = 43,350 DOF | 同上 |
| `s3_brick_transient.tcl` | 3D 实体块瞬态（Newmark 200 步） | 10×10×40 = 14,883 DOF | 同上 |

设计要点（面试能讲）：
- **场景多样性**：S1 是大规模 SPD（测 CG 优势）、S2 是带梁单元的框架（对直接法 fill-in 病态）、S3 是瞬态（测「每步一次求解」的累积效应）；
- **统一 RESULT 输出**：方便脚本化解析、自动化汇总。

## 2.3 先校准（小规模）再全量（大规模）

AI 没有一上来就 18×18×200 跑全量，而是**先用小规模验证三条求解链路都通**：
```bash
# 串行 SuperLU 小规模
../build/OpenSees models/s1_brick_static.tcl superlu 4 4 20
# CUDA 小规模
LD_PRELOAD=... ../build-cuda/OpenSees models/s1_brick_static.tcl cusp-cg 4 4 20
# 并行 Mumps 小规模（带 timeout 防卡死）
timeout 120 mpiexec -np 4 ../build/OpenSeesSP models/s1_brick_static.tcl mumps 4 4 20
```
确认小规模三路都出正确 RESULT 后，才逐步放大到 15×15×160 → 18×18×200。

**这是工程上的「冒烟测试 → 全量」纪律**：先证明链路通，再投时间跑贵的大规模，避免跑一晚发现脚本有 typo。

## 2.4 写自动化脚本，后台跑全量

```bash
# run_all.sh    —— 把所有 (场景 × 求解器 × 规模) 组合串起来，tee 到日志
# summarize.py  —— 从 RESULT 行汇总成表格
# calibrate.sh  —— 规模校准记录
bash run_all.sh 2>&1 | tee results/run_all_console.log   # 后台过夜跑
```

## 2.5 正确性验证：不只是比 tipDisp，是比整个位移场

你问「S2 187× 好快，结果是否对得上」——AI 没有只信 tipDisp，而是写了 `verify_field.tcl` / `verify_frame_field.tcl`，把**全部节点位移**从 superlu 和 cusp 各 dump 一份，逐分量对比：

```python
a = load('disp_superlu.out'); b = load('disp_cusp.out')
diffs = [abs(x-y) for x, y in zip(a, b)]
print(f'最大相对差: {max(diffs)/max(abs(a)):.3e}')   # → 位移场级一致
```

**这比「tipDisp 对上」强得多**：tipDisp 只验证一个点，逐分量对比验证整个解向量。面试讲这个，说明你懂「正确性验证要覆盖整个解，不是抽查一个数」。

## 2.6 Mumps 并行排错（phase A 最有价值的一段）

首轮结果 Mumps 很糟：
- S1 np=4 → **Mumps Error -9（每进程内存不足）**
- S2 → np=4/np=8 都 **30 分钟超时**

你连着追问「为什么 np=8 就能跑、np=4 不够」「我 32GB 内存」「为什么串行反而比并行快」。

**AI 的定位链**（这就是「定位性能问题」的方法论）：

1. **Error -9 根因**：MUMPS 是**分布式内存**并行，LU 因子按进程分摊。15GB WSL 内存 ÷ np 个进程，np=4 时单进程份额翻倍、超了 15GB；np=8 份额减半才放得下。所以「内存够」是总量够，但**单进程分摊量**超了。

2. **S2 超时根因**：查了 `nm -u libmumps*.a | grep -i metis`，发现这个 MUMPS 构建**只带 PORD 排序**（一个很弱的排序器），没有 METIS/SCOTCH。S2 框架模型 fill-in 本来就大，PORD 排序下 fill-in 爆炸 → 直接法卡死。

3. **修复**：重编一个带 **SCOTCH 排序**的 MUMPS（`build-metis`），再链接进 OpenSees 的 `build-hypersp`，中间还踩了两个链接坑：
   - `MUMPS_LIBRARIES` 要补 `libesmumps.a`（SCOTCH 的 ESMUMPS 接口）；
   - `SCALAPACK_LIBRARIES` 要显式指到 `/usr/lib/.../libscalapack-openmpi.so`。

4. **结果翻转**：换 SCOTCH 排序后，S2 从「30 分钟超时」变成 np=4 4.61s、np=8 3.72s；S1 np=8 37.5s。**排序器对直接法的性能是决定性的**——这是 phase A 报告第五节「第二轮」的由来。

> 面试金句：「首轮 Mumps 慢，我先怀疑是并行直接法本身不行；但查了链接符号发现它根本没带 METIS/SCOTCH，只用了最弱的 PORD 排序。换 SCOTCH 重编后 S2 从超时变成 4 秒——**瓶颈不是算法，是配置**。」

## 2.7 升级 CUDA 12.8（为 phase B 做准备）

你问「现在 cuda 版本是什么」→ 12.0，不支持 sm_120。你拍板「升级 WSL 里 Toolkit 到 12.8」。

AI 的操作（关键命令）：
```bash
curl -fSL -o ~/cuda-install/cuda_12.8.1_linux.run \
  https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8 --no-driver
# 验证 sm_120 支持
~/cuda-12.8/bin/nvcc --list-gpu-arch | grep 120   # → sm_120 出现了
```

要点：
- `--toolkitpath=$HOME/cuda-12.8` 装到用户目录，**不动系统**，和老的 12.0 共存；
- `--no-driver`：WSL 里驱动是 Windows 侧的，**不装 Linux 驱动**（这正是坑 4 的教训）。

## 2.8 会话二收尾：启动 phase B 的 ILU0 空壳

AI 把 `CuSPSolver` 里的 ILU0 两个空壳填上了（`factorizeILU0()` 用 `csrilu02` 分解、`applyPreconditioner()` 用 `SpSV` 前推/后推），头文件补了资源成员——**注意，这时候用的是 `csrilu02` + `SpSV`，正是会话三要踩的两个坑**。

---

# 会话三：Phase B ILU0 + sm_120 坏 API 排查（8/11–8/12）

你的起点消息：

> 当前在阶段 B 第 2 步：nvcc 12.8 单独语法检查 CuSPSolver.cpp … 正在补 include 路径重试

## 3.1 先学会一招：nvcc 单独语法检查（快，不用全量编译）

全量 `make` 太慢，AI 用 nvcc **单独编译一个 .cpp** 来快速查语法：

```bash
FLAGS=~/OpenSees/build-cuda/CMakeFiles/OPS_SysOfEqn.dir/flags.make
INC=$(grep -m1 "^CXX_INCLUDES" $FLAGS | sed 's/^CXX_INCLUDES = //')   # 240 个 -I
cd ~/OpenSees/SRC/system_of_eqn/linearSOE/sparseGEN && \
  ~/cuda-12.8/bin/nvcc -x cu -c CuSPSolver.cpp -o /tmp/CuSPSolver_check.o \
  $INC -std=c++17 -gencode arch=compute_120,code=sm_120
```

两个关键点（面试能讲）：
1. **`.cpp` 必须加 `-x cu`**：因为文件里有 `<<<>>>` kernel 启动语法，普通 C++ 编译器（或 nvcc 的默认 .cpp 模式）不认识；
2. **include 路径从 `flags.make` 的 `CXX_INCLUDES` 提取**：这 240 个 `-I` 是 CMake 攒出来的完整依赖，直接复用，不用手动猜。

修了几个小错误：
- `cusparseSpSV_solve` 是 **9 参**（外部缓冲只在 `analysis` 传，solve 不传），初版多传了 buffer → 编译错；
- BiCGStab 里多声明的 `dot_rr` 变量删掉（你提醒过「先改这个再去编译」）。

## 3.2 崩溃：S1 大模型 ilu0 配 SIGFPE

`cusp-ilu0-bicgstab 18 18 200` 一跑就崩，gdb 栈顶：

```
cusparseSpSV_solve ← applyPreconditioner ← solveCG
```

AI 没有立刻改代码，而是**写独立复现器**（这是整个 phase B 最值钱的方法论）：

## 3.3 独立复现器 `spsv_repro.cu`：把问题从 OpenSees 里剥离

```bash
# 一个 7 点 Laplacian，复刻 csrilu02 → SpSV 调用链，3 个 mode：
#   mode 0 = fresh 描述符   1 = 复用描述符   2 = 跳过 ILU0 直接测 SpSV
~/cuda-12.8/bin/nvcc -x cu spsv_repro.cu -o spsv_repro \
  -I$HOME/cuda-12.8/include -L$HOME/cuda-12.8/lib64 -lcusparse \
  -gencode arch=compute_120,code=sm_120
./spsv_repro 60 0   # n=60³ 附近必现 FPE
```

**二分定位**：
- n≥27,000 必现 FPE，n=1,000 正常 → **规模相关**；
- 复用/新建描述符都崩 → 与描述符生命周期无关；
- **跳过 ILU0 直接 SpSV 正常** → 问题在 `csrilu02`，不在 SpSV；
- **完整性检查实锤**：分解后把结构数组拷回 CPU 与原始结构 `memcmp`，发现 **170 万个索引被改写、对角线变 0** —— `csrilu02` 在 sm_120 大矩阵上**越界写**传入的 rowPtr/colInd，状态码却还报成功。

> 面试金句：「崩溃栈指向 SpSV，但我不直接改 SpSV——先写最小复现器，二分定位到 `csrilu02` 越界写结构数组（memcmp 实锤 170 万索引被改），再决定怎么绕。**问题不是『看起来崩在哪』，是『实测是哪个库在写坏内存』。**」

## 3.4 顺带修掉 WSL libcuda（会话一坑 4 的彻底版）

修 libcuda 时 AI 发现只删软链没用：
- `ldconfig` 会按 ELF soname **自动重建断链**；
- 必须**移走实体文件**，不能 `apt purge libnvidia-compute-*`（模拟显示 apt 会换装 `-server` 变体驱动，问题照旧）。

最终修复（已执行）：
```bash
sudo mv /usr/lib/x86_64-linux-gnu/libcuda.so.580.173.02.disabled /root/
sudo rm -f /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
sudo ldconfig
# 之后 ldconfig -p 只剩 /usr/lib/wsl/lib，OpenSees 不带 LD_PRELOAD 直接跑通
```

## 3.5 修 csrilu02：先加「完整性自检 + 干净回退」，再换 CPU 分解

分两步（先止血、后根治）：

**第一步（止血）**：ILU0 全程用**专用结构副本** `d_rowPtrLU/d_colIndLU`（不让 csrilu02 碰 SpMV 共用的 `d_csrRowPtr/d_csrColInd`）；分解后把副本拷回 CPU 与原始结构 `memcmp`，不一致 → 打印警告 + `m_factorized=false` + **干净回退 Jacobi**。这样「隐性崩溃」变成「显式降级」。

**第二步（根治）**：既然 `csrilu02` 在 sm_120 上不可信，直接**弃用**，把 ILU0 分解改到 **CPU 端等价实现**（稠密工作区 `w[]` + `marker[]`，即 `learning_guide.md` 3.1 讲的 scatter/eliminate/gather），分解结果再上传 GPU。CPU ILU0 逻辑简单可控，不受 GPU 库 bug 影响。

## 3.6 修 SpSV：也弃用，自实现 level-scheduled 三角求解

CPU ILU0 替换后，崩溃率从 7/10 降到 1/10——**还剩 1/10 的崩溃栈仍指向 `cusparseSpSV_solve`**，说明 **SpSV 在 sm_120 上也间歇性 SIGFPE**（和 csrilu02 是同一个平台病根）。

于是彻底弃用 SpSV，自实现三角求解：
- 头文件删掉 SpSV/csrilu02 全部成员，新增 level 调度成员（`d_fwdOrder/d_bwdOrder/d_fwdLevelPtr/d_bwdLevelPtr/m_numFwdLevels/m_numBwdLevels`）和 CPU ILU0 工作区；
- `factorizeILU0()`：CPU 等价 ILU(0) 后，首次构建前推/后推 **level 调度**（前推 `level[i]=max(level[j]+1, j<i)`，后推从后往前同理），桶排成 `fwdOrder/bwdOrder` 上传 GPU，**只依赖结构、只做一次**；
- `applyPreconditioner()` ILU0 分支：逐 level 启动 `fwdSubKernel`（单位对角前推 L·w=b）+ `bwdSubKernel`（非单位对角后推 U·x=w）。

> 这一步的数学原理见 `learning_guide.md` 3.4（level scheduling：依赖 DAG 分层 → 同层并行 + 层间 `grid.sync()`）。

## 3.7 瞬态 FPE 的二分定位

S3 瞬态（200 步）也会崩，AI 用**步数二分**把崩溃压到最前：
```bash
for ns in 1 5 20 50; do ... cusp-ilu0-cg 10 10 40 $ns ...; done
# 找到「steps≥2 就 FPE」的最小复现
```
（因为 gdb 托管跑 CUDA 极慢，200 步瞬态等不起，先用步数二分把崩溃压到 steps=1 的规模再 gdb / compute-sanitizer。）

## 3.8 验证通过，收尾

最终验证矩阵（对应 `phaseB_benchmark_log.md`）：
- 8×8×40 连续 10 次 → 10/10，tipDisp 逐位一致；
- S1 全尺寸 217,683 DOF（csrilu02 原必崩规模）→ 无崩溃、无 zero pivot、无回退；
- S3 瞬态 20/200 步 → 通过，与 superlu 偏差 ≤1.3e-9；
- 三场景 ilu0-cg / ilu0-bicgstab 正确性 → 与 superlu 基线一致。

随后写 `phaseB_implementation_log.md` 和 `phaseB_benchmark_log.md`。

---

# 把「操作」和「原理」对上（学习路径建议）

| 你想学的 | 看哪份 | 重点 |
|---|---|---|
| 每个 bug 是怎么一步步定位的 | 本文档会话一/三 | gdb 断点、最小复现器、memcmp 完整性检查、步数二分 |
| CSR 三数组、CG/Jacobi、ILU0、BiCGStab 的数学 | `learning_guide.md` §1–3 | 面试最爱问的原理 |
| level scheduling 为什么慢、瓶颈怎么定位 | `learning_guide.md` §4 + `phaseB_benchmark_log.md` §5 | 「假设→实验→定位」叙事 |
| 每个配置的精确数字 | `phaseA_benchmark_log.md` / `phaseB_benchmark_log.md` | 加速比、tipDisp、偏差 |
| 怎么亲手跑一遍 | `learning_guide.md` §5（含 §5.5 实测复现记录） | 复现命令 + 预期输出 |

**建议的学路径**：先读本文档了解「全过程发生了什么」，再对着 `learning_guide.md` 理解「为什么这么做」，最后用 §5 的命令自己跑一遍——三份合起来就是你能完整复述、能答辩的 phase A/B 故事。
