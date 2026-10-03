# 阶段 A 实施工作记录：cuSPARSE GPU 求解器跑通

> 日期：2026-08-02
> 目标：在 OpenSees 中接入自研 cuSPARSE 迭代求解器（`CuSPSolver`），完成阶段 A（CG/BiCGStab + Jacobi 预条件，正确性验证）
> 结果：**完成**。GPU 求解结果与 CPU（SuperLU）及解析解三方吻合

---

## 一、最终验证结果

测试模型：`test_cusp.tcl`（10 单元 2D 悬臂梁，端部竖向集中力 1000，L=100, E=29000, I=1000）

| 求解器 | 端部挠度 dy | 与解析解相对偏差 |
|---|---|---|
| 解析解 P·L³/(3EI) | -11.494252873563218 | — |
| SuperLU (CPU) | -11.494252873552977 | ~9e-12 |
| **cusp CG + Jacobi** | -11.494252873559686 | ~3e-12 |
| **cusp BiCGStab + Jacobi** | -11.494252873937057 | ~3e-11 |

运行方式（WSL libcuda 冲突未修复前）：

```bash
cd ~/OpenSees
./build-cuda/OpenSees test_cusp.tcl superlu                    # CPU 基准
LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1 ./build-cuda/OpenSees test_cusp.tcl cusp   # GPU
```

Tcl 用法：`system cusp -pre jacobi -solver cg -rtol 1e-10 -maxIter 10000`
（`-pre`: none/jacobi/ilu0；`-solver`: cg/bicgstab；命令名 `cusp`、`cuSPARSE` 均可）

---

## 二、代码主要修改

### 2.1 `CMakeLists.txt`（根目录）

CUDA 检测块（约 43-66 行）：

- `enable_language(CUDA)` —— 启用 CUDA 编译语言（**之前缺失，是 nvcc 不生效的根因**）
- `CMAKE_CUDA_ARCHITECTURES "75;80;89;90;90-virtual"` —— 89 原生覆盖 4090 生产环境；90-virtual 生成 PTX 供 RTX 5070（sm_120）驱动 JIT（CUDA 12.0 不认识 sm_120）
- `CMAKE_CUDA_RUNTIME_LIBRARY Shared` —— CUDA 运行时用共享库（默认 Static 会引发段错误，见 3.4）
- 全局 `include_directories(sparseGEN)` —— 让 `commands.cpp`/`solver.cpp` 能找到 `CuSPSolver.h`
- `-static-libgcc -static-libstdc++` 改为**仅非 CUDA 构建**生效（原 105 行，用 `if(NOT CUDAToolkit_FOUND)` 包裹）
- `OpenSees` 目标：`target_compile_definitions(OpenSees PRIVATE _CUSP)` + 链接 `CUDA::cudart/cusparse/cublas`（位置在 `OpenSeesLIB` 之后，保证静态库符号解析顺序）

### 2.2 `SRC/system_of_eqn/linearSOE/sparseGEN/CMakeLists.txt`

```cmake
set_source_files_properties(CuSPSolver.cpp
    TARGET_DIRECTORY OPS_SysOfEqn
    PROPERTIES LANGUAGE CUDA)
```

**注意**：`TARGET_DIRECTORY` 是独立参数，写进 `PROPERTIES` 里会被当成属性名静默忽略（见 3.2）。

### 2.3 `SRC/system_of_eqn/linearSOE/sparseGEN/CuSPSolver.h / .cpp`

- 整体重写：CG / BiCGStab 迭代循环 + Jacobi 预条件（3 个自定义 kernel：提取对角线、求逆、逐元素乘），cuBLAS 做点积/axpy/nrm2
- **API 迁移**：CUDA 12.0 已删除全部 legacy cuSPARSE API（`cusparseDcsrmv`、`csrsv2` 等），SpMV 改用 generic `cusparseSpMV`：
  - 新增静态辅助函数 `gpuSpMV()`（每次调用临时创建 DnVec 描述符）
  - 新增成员 `cusparseSpMatDescr_t m_spMatA`（CSR 描述符，首次 solve 时创建、复用）、`void *d_spmvBuffer`、`size_t m_spmvBufferSize`
- 删除 CUDA 12 不存在的 `cusparseSolveAnalysisInfo_t` 成员；`csrilu02Info_t` 保留（阶段 B 用）
- 新增状态标志 `m_structureUploaded`（修复结构上传 bug，见 3.6）
- 构造函数先调 `cudaGetDeviceCount()` 再 `cusparseCreate()`（规避 Blackwell bug，见 3.7），并增加错误检查输出

### 2.4 `SRC/tcl/commands.cpp`（旧注册系统 —— tcl exe 实际使用的）

- 移植 `#ifdef _CUSP` 块到 Linux：`_stricmp` → `strcasecmp`（加 `#include <strings.h>`）
- 参数编号与新构造函数对齐：solver `cg=0/bicgstab=1/gmres=2`（旧代码是 bicg=0/bicgstab=1/cg=2，**编号不一致会导致选错求解器**）；preCond `none=0/jacobi=1/ilu0=2`（旧代码的 ainv 已不存在）
- 兼容 `system cusp` 与 `system cuSPARSE` 两种命令名；`-maxIter` 与旧式 `-mInt` 均可

### 2.5 `SRC/runtime/commands/CMakeLists.txt` + `analysis/solver.cpp/hpp`

- `OPS_Runtime` 目标加 `_CUSP`（新系统 `soe_table` 注册保留，供 OpenSeesPy 使用；对 tcl exe 无效，见 3.5）

### 2.6 `test_cusp.tcl`（新增）

对比测试脚本，用法：`OpenSees test_cusp.tcl [cusp|superlu]`，打印数值解与解析解。

---

## 三、Bug 修理记录（按发现顺序）

### 3.1 nvcc 不参与编译

- **症状**：`blockIdx was not declared`、`expected primary-expression before '<'`
- **根因**：`sparseGEN/CMakeLists.txt` 设了 `LANGUAGE CUDA` 属性，但根 CMake 从未 `enable_language(CUDA)`，属性不生效，g++ 编译含 kernel 语法的 .cpp
- **修复**：根 CMake CUDA 块补 `enable_language(CUDA)`

### 3.2 `TARGET_DIRECTORY` 位置错误

- **症状**：补了 `enable_language` 后仍显示 "Building CXX object CuSPSolver.cpp.o"
- **根因**：`OPS_SysOfEqn` 在上级目录创建，`set_source_files_properties` 的属性默认只在本目录作用域生效；且 `TARGET_DIRECTORY` 被误写进 `PROPERTIES` 列表里，被当成属性名静默忽略
- **修复**：`set_source_files_properties(CuSPSolver.cpp TARGET_DIRECTORY OPS_SysOfEqn PROPERTIES LANGUAGE CUDA)`

### 3.3 `_CUSP` 宏作用域

- **症状 1**：只对 `OPS_SysOfEqn` 定义时，注册代码（`solver.cpp`/`commands.cpp`）被静默编译掉，`system cusp` 运行时不存在
- **症状 2**：改为全局定义后，`commands.cpp` 里旧的 Windows-only `_CUSP` 块被激活，`_stricmp` 未定义 → 编译失败
- **修复**：`_CUSP` 只定义在三个目标上：`OPS_SysOfEqn`（求解器本体）、`OPS_Runtime`（新系统注册）、`OpenSees`（旧系统注册）；同时把旧块移植到 Linux（见 2.4）

### 3.4 启动即段错误（本次最疑难的 bug）

- **症状**：CUDA 版 `OpenSees` 连 `puts "hello"` 都崩；无 CUDA 版正常
- **定位过程**：装 gdb → 调用栈显示崩在 `std::ostream::sentry`（`puts` → `opserr` → `cerr << s`）→ 检查发现在 `main()` 时 exe 内 `std::cerr` 的 **vtable 为 NULL**
- **根因**：根 CMake 无条件设置 `-static-libstdc++`（105 行，OpenSees 为二进制分发所设），exe 静态链接 libstdc++ 后有自己一份 `std::cerr`；与 CUDA 链接布局叠加后该对象的初始化被扰乱（vtable 保持 NULL），第一次流输出即崩
- **修复**：CUDA 构建禁用 `-static-libstdc++`（CUDA 二进制本就依赖 CUDA 运行时，静态打包无意义），共享 libstdc++.so 的 cerr 由 .so 自行初始化

### 3.5 注册系统误判（架构纠正）

- **症状**：修复编译后，`system cusp` 在 tcl exe 中不存在
- **根因**：`.plan.md` 假设"当前版本使用新系统（soe_table）"，但实测 `OpenSees` tcl exe **并未链接 `OPS_Runtime`**（link.txt 只有 `OPS_InterpTcl` + `OpenSeesLIB`），`system` 命令由旧系统 `commands.cpp` 处理；新系统 `solver.cpp` 只服务于 OpenSeesPy
- **修复**：移植旧系统 `_CUSP` 注册块；新系统注册保留给 Python 接口

### 3.6 CSR 结构从未上传 GPU

- **症状**：修复注册后仍崩在 `cublasDdot`，cusparse 报 `csrRowOffsets NULL`
- **根因**：OpenSees 先调 `setSize()`（内部调 `allocateGPU()` 把 `m_initialized` 置 true），再调 `solve()`；而 `solve()` 用 `if (!m_initialized)` 判断"首次调用"，导致 CSR 行指针/列索引拷贝和描述符创建**永远被跳过**，`d_csrRowPtr` 是垃圾、`m_spMatA` 是 nullptr
- **修复**：新增独立标志 `m_structureUploaded`，首次 `solve()` 时上传结构 + 创建描述符；`freeGPU()` 中同步重置并销毁描述符

### 3.7 Blackwell 上 cusparseCreate 首调失败

- **症状**：`cusparseCreate(): CUDA context cannot be initialized`，status=1，handle=NULL
- **根因**：CUDA 12.0 已知 bug——RTX 5070（sm_120）上若 `cusparseCreate` 是进程第一个 CUDA 调用，驱动侧上下文初始化失败；先用任何运行时 API 初始化即正常（独立探针程序复现验证）
- **修复**：构造函数中先调 `cudaGetDeviceCount()` 再 `cusparseCreate()`

### 3.8 WSL libcuda 冲突（环境问题，待用户 sudo 处理）

- **症状**：修复 3.7 后报 "no CUDA-capable device detected"
- **根因**：`/lib/x86_64-linux-gnu/libcuda.so.580.173.02`（6/29 手动安装的 Linux 原生驱动库，非 dpkg 管理）抢占加载；WSL 中 GPU 只能通过 `/usr/lib/wsl/lib/libcuda.so.1`（转发 Windows 驱动的 shim）访问，原生驱动枚举不到 GPU
- **临时方案**：`LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1 ./build-cuda/OpenSees ...`（已验证有效）
- **永久方案**（需用户执行）：
  ```bash
  sudo mv /lib/x86_64-linux-gnu/libcuda.so.580.173.02 \
          /lib/x86_64-linux-gnu/libcuda.so.580.173.02.disabled && sudo ldconfig
  ```
- 注：此问题仅存在于本 WSL 开发机，生产环境 4090 服务器（原生 Linux）不受影响

---

## 四、阶段 B 待办（ILU0 预条件）

- ILU0 分解：`cusparseDcsrilu02`（CUDA 12 尚存，已 deprecated）
- 三角求解：`csrsv2` 已从 CUDA 12 删除，**必须**用 generic `cusparseSpSV`（createDescr/analysis/solve），L、U 各一套描述符，用 `cusparseFillMode_t` + `cusparseDiagType_t` 属性区分
- 建议先升级 CUDA Toolkit 到 12.8+：原生支持 sm_120（`CMAKE_CUDA_ARCHITECTURES` 改 `"89;120"`），generic API 一次写到位，避免在 12.0 上写 deprecated API 再返工
- `factorizeILU0()` / `applyPreconditioner()` 的 ILU0 分支目前是空壳 + Jacobi 兜底，代码位置已标注 TODO
