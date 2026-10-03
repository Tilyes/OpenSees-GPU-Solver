# 阶段 B 工作日志：ILU0 预条件实现与 sm_120 平台问题排查

> 日期：2026-08-11 ~ 2026-08-12
> 目标：填完 CuSPSolver 的 ILU0 预条件（factorizeILU0 + applyPreconditioner SpSV 前推/后推），通过编译与功能验证
> 状态：**完成**。ILU0 + BiCGStab 全流程通过编译与三场景功能验证；csrilu02 与 SpSV 两个坏 API 均已自实现替换，sm_120 间歇 SIGFPE 消除（见第五节）。性能瓶颈已定位 = level 调度临界路径 O(n^{2/3})，非 launch 开销（见第六节）

---

## 一、ILU0 代码实现（已完成）

- `factorizeILU0()`：`cusparseDcsrilu02_bufferSize/analysis/numeric`（CUDA 12 唯一 ILU0 入口，已 deprecated 但可用），结果存 `d_csrValLU`
- `applyPreconditioner()`：ILU0 分支用 generic SpSV API 做前推 `L*w=b`（单位对角）+ 后推 `U*x=w`（非单位对角），中间向量专用 `d_w`（避免与调用方缓冲别名）
- 头文件新增资源成员：`d_csrValLU / d_rowPtrLU / d_colIndLU / d_iluBuffer / m_spMatL / m_spMatU / m_spsvL / m_spsvU / d_spsvBufL/U / d_w`，构造/析构/freeGPU 同步
- **API 修正**：CUDA 12.8 的 `cusparseSpSV_solve` 是 9 参（外部缓冲只在 `analysis` 传），初版多传了 buffer 参数导致编译错误
- **nvcc 单独语法检查方法**：`.cpp` 必须加 `-x cu` 否则 kernel 语法不识别；include 列表从 `build-cuda/CMakeFiles/OPS_SysOfEqn.dir/flags.make` 的 `CXX_INCLUDES` 提取（240 个 `-I`）

## 二、问题 1：csrilu02 写坏 CSR 结构数组（已修复）

**现象**：S1 大模型（217,683 DOF）ilu0 配置 SIGFPE 崩溃，gdb 栈顶 `cusparseSpSV_solve ← applyPreconditioner`。

**定位**：独立复现器 `plan/spsv_fpe_repro.cu`（7 点 Laplacian，复刻 ILU0+SpSV 调用链）：
- n≥27,000 必现 FPE，n=1,000 正常
- 复用/新建 DnVec 描述符都会崩 → 与描述符生命周期无关
- 跳过 ILU0 直接 SpSV 正常 → 问题在 csrilu02
- **完整性检查实锤**：分解后拷回结构数组比对，170 万个索引被改写、对角线变 0——`csrilu02` 在 sm_120 大矩阵上会越界写传入的 rowPtr/colInd，且状态码仍报成功

**修复**（`CuSPSolver.cpp`，已验证）：
1. ILU0 全程使用**专用结构副本** `d_rowPtrLU/d_colIndLU`（结构上传时分配填充），不碰 SpMV 共用的 `d_csrRowPtr/d_csrColInd`
2. 分解后把副本 D2H 与 CPU 原结构 `memcmp` 比对，不一致 → 打印警告、`m_factorized=false`、**干净回退 Jacobi**
3. 换用副本后越界未再复现（原崩溃疑似与越界目标的显存布局相关）

**验证**（libcuda 修复前环境，均无需回退）：

| 场景 | 配置 | 时间 | tip 位移 | 与基线偏差 |
|---|---|---|---|---|
| S1 静力 217,683 DOF | cusp-cg | 4.37s | 0.3282677550444 | 基线 |
| S1 | cusp-ilu0-cg | 6.41s | 0.3282677550223 | ~2e-11 |
| S1 | cusp-ilu0-bicgstab | 12.55s | 0.3282677551594 | ~1e-10 |
| S2 框架 43,350 DOF | cusp-cg | 0.60s | 52.816740539856 | 基线 |
| S2 | cusp-ilu0-cg | 1.22s | 52.816740539857 | ~1e-14 |
| S2 | cusp-ilu0-bicgstab | 1.92s | 52.816740539858 | ~4e-14 |
| test_cusp 悬臂梁 | ilu0+bicgstab | — | -11.494252873567 | 解析解 -11.494252873563 |

## 三、问题 2：WSL libcuda 环境损坏（已修复）

**现象**：不带 `LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1` 就 `cusparseCreate failed`；compute-sanitizer 与 LD_PRELOAD 互斥无法使用。

**根因**：WSL 里装过 Linux 版驱动包 `libnvidia-compute-580`（+残留 535）。此前有人把实体库改名为 `libcuda.so.580.173.02.disabled`，但：
- 软链 `libcuda.so.1` 仍指向 disabled 文件 → 加载即坏
- **`ldconfig` 会按 ELF soname 自动重建断链**（实测 09:26 被重新生成）→ 只删软链没用，必须移走实体文件

**修复**（需 sudo，已执行并验证）：
```bash
sudo mv /usr/lib/x86_64-linux-gnu/libcuda.so.580.173.02.disabled /root/
sudo rm -f /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
sudo ldconfig
```
注意：**不要** `apt purge libnvidia-compute-*`——模拟显示 apt 会换装 `-server` 变体驱动，同样是 Linux 驱动，问题照旧。

**修复后**：`ldconfig -p` 只剩 `/usr/lib/wsl/lib`；OpenSees 不带 LD_PRELOAD 直接跑通；compute-sanitizer 恢复可用。

## 四、排查方法备忘（坑与技巧）

- WSL 下 compute-sanitizer 不能配 LD_PRELOAD（注入冲突）；修好 libcuda 后两者都不需要
- gdb 托管跑 CUDA 极慢，200 步瞬态等不起 → 先用步数二分（`steps=1/5/20/50`）把崩溃压到最前
- 批量跑 GPU 算例**不要并发**：两个 OpenSees 进程共享 GPU 会干扰结论（一次 200 步批跑卡死 11 分钟被终止，原因未完全排除并发因素）
- 后台任务的审批弹窗不会一直挂着等人：会话空闲时任务完成通知可能延迟到下次活跃才处理， overnight 任务要一次批完

## 五、问题 3：SpSV 间歇 SIGFPE → 自实现 level-scheduled 三角求解（已解决）

CPU ILU0 替换 csrilu02 后，崩溃率从 7/10 降到 1/10。剩余 1/10 的崩溃栈仍指向 `cusparseSpSV_solve ← applyPreconditioner ← solveCG`——即 **cusparseSpSV 在 sm_120 上同样间歇性 SIGFPE**（与 csrilu02 是同一平台病根的又一表现）。

**修复**（`CuSPSolver.{h,cpp}`，已验证）：彻底弃用 SpSV，改为自实现 level-scheduled 三角求解：

1. 头文件删除 SpSV/csrilu02 相关成员（`m_infoILU/d_iluBuffer/m_spMatL/m_spMatU/m_spsvL/m_spsvU/d_spsvBuf*/m_spsvBufSize*`），新增 level 调度成员（`d_fwdOrder/d_bwdOrder/m_fwdLevelPtr/m_bwdLevelPtr/m_numFwdLevels/m_numBwdLevels/m_levelsBuilt`）与 CPU ILU0 工作区（`h_valLU/h_w/h_diagU/h_marker`）
2. `factorizeILU0()`：CPU 端等价 ILU(0)（稠密 w + marker），分解后首次构建前推/后推 level 调度（前推 `level[i]=max(level[j]+1, j<i)`，后推从后往前同理），桶排成 `fwdOrder/bwdOrder` 上传 GPU，只依赖结构、做一次
3. `applyPreconditioner()` ILU0 分支：逐 level 启动 `fwdSubKernel`（单位对角前推 L·w=b）与 `bwdSubKernel`（非单位对角后推 U·x=w），中间向量用专用 `d_w`（避免与调用方 `d_tmp` 别名）

**验证**：

| 项 | 结果 |
|---|---|
| 8×8×40 连续 10 次 | ✅ 10/10 通过，tipDisp 完全一致 |
| S1 全尺寸 217,683 DOF（csrilu02 原必崩规模） | ✅ 无崩溃、无 zero pivot、无回退 |
| S3 瞬态 20 / 200 步（原 steps≥2 FPE） | ✅ 通过，tip 位移与 superlu 偏差 ≤1.3e-9 |
| 三场景 ilu0-cg / ilu0-bicgstab 正确性 | ✅ 与 superlu 基线一致（详见 `phaseB_benchmark_log.md`） |

## 六、性能瓶颈定位（persistent kernel 实验 → 临界路径）

Level-scheduled 三角求解初版是「每个 level 一个 kernel」，786/~6972 层 = 几百到几千次 launch。怀疑 launch 开销是主因，做了两步实验：

1. **persistent 单 block**：level 循环搬进 kernel → 8×8×40 仅 2.237s→2.025s（~10%），launch 开销不是主导；
2. **cooperative persistent**（全网格常驻 + 逐 level `grid.sync()`）→ S1 51.79s→51.27s（无改善）、小规模反慢，进一步排除 launch/带宽。

**结论**：真正瓶颈是 level 调度的依赖临界路径本身（RCM 下 O(n^{2/3})，实测 786/~6972 层），不是 launch 开销。ILU0 把 CG 迭代压到 2.3×（94 vs 219）但每次 apply 是 Jacobi 的 ~25×，红利被抵消。完整量化表见 `phaseB_benchmark_log.md` 第五节。

**遗留尝试（已回退）**：曾试单源 BFS 置换把临界路径压向 O(diameter)，实测只降 786→687（~12%）——stdBrick 单元耦合对角节点、同一 BFS 壳层内仍留长链。代码已回退到 cooperative persistent 版（8×8×40 连续 10/10 通过）；置换重排作为展望方向 A 保留（见 benchmark 报告第六节结论）。

## 七、待办

1. ~~S3 瞬态 FPE 定位与修复~~ → 已解决（SpSV 自实现替换）
2. 4090（sm_89）真机验证 ILU0 大模型——构建已含 `89` 架构（`CMakeLists.txt:50`），部署后直接跑 `bench/models/s1_brick_static_ilu0.tcl`
3. ILU0 性能优化（瓶颈已定位 = 临界路径，见第六节；展望方向 A/B/C 见 `phaseB_benchmark_log.md` 第六节结论，置换重排已回退待后续实现）
4. 非对称/病态场景验证 ILU0/BiCGStab 的收敛加速价值（当前三场景均 SPD）

## 附：相关文件

| 文件 | 说明 |
|---|---|
| `plan/spsv_fpe_repro.cu` | SpSV FPE 独立复现器（用法：`./spsv_repro <nx> <mode>`，mode 0=fresh描述符 1=复用 2=跳过ILU0） |
| `plan/learning_guide.md` | **学习指南**：phase A/B 的完整原理讲解 + 代码走读 + 操作复现 + 面试追问（想学会细节先读这个） |
| `plan/operation_history.md` | **操作全过程还原**：从 Kimi Code 会话记录还原的每一步操作（命令/改文件/bug 定位链路），与 learning_guide 互补 |
| `bench/models/s1_brick_static_ilu0.tcl` | S1 模型 + `cusp-ilu0-cg` / `cusp-ilu0-bicgstab` 两个配置 |
| `SRC/system_of_eqn/linearSOE/sparseGEN/CuSPSolver.{h,cpp}` | ILU0 实现 + 结构副本 + 完整性自检 |
