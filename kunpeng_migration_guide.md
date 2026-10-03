# OpenSeesSP 鲲鹏服务器移植指南（无 root 权限版）

> **场景**：你在 WSL (Ubuntu 24, x86_64) 上已成功编译 OpenSeesSP，现在要移植到鲲鹏服务器（openEuler, aarch64），但**没有服务器的完整操控权限（无 sudo/root）**。
>
> **策略**：分两阶段执行
> - **阶段一（本地准备，在 WSL 上做）**：不需要服务器权限就能完成的所有工作 —— 源码整理、架构兼容性检查、依赖源码准备、构建脚本编写、传输打包
> - **阶段二（服务器有限权限部署）**：只用用户空间工具（Miniconda + 源码编译），所有文件安装在 `$HOME` 下，不需要 root

---

## 当前构建环境参考 (WSL x86_64)

| 项目 | 当前配置 |
|---|---|
| 编译器 | GCC 13 (`gcc` / `g++` / `gfortran`) |
| 构建系统 | CMake |
| 目标 | `OpenSeesSP` |
| MPI | OpenMPI (系统包) |
| BLAS/LAPACK | 系统库 `libblas.so` |
| MUMPS | 独立编译于 `/home/jumjum/mumps/build`，通过 `-DMUMPS_DIR` 传入 |
| HDF5 | 系统包 (1.10.10) |
| TCL | 系统包 (8.6) |
| Python | 系统包 (3.12) |
| Eigen3 | 系统包 |
| MKL / CUDA | 未使用（对迁移有利） |

### CMake 关键变量（已从源码验证）

以下变量名直接来自 `CMakeLists.txt`，**必须完全匹配**：

| 变量名 | 作用 | 源码位置 |
|---|---|---|
| `MUMPS_DIR` | MUMPS 编译产物目录，不定义则禁用 MUMPS (`_NOMUMPS`) | 第 435 行 |
| `SCALAPACK_LIBRARIES` | ScaLAPACK 库列表，用分号分隔 | 第 467-487 行 |
| `OPENMPI` | 定义后启用 `_OPENMPI` 宏 | 第 744 行 |
| `LAPACK_LIBRARIES` | LAPACK 库（可选，不传则自动查找或用内置源码） | 第 389 行 |

> ⚠️ **不存在** `OPEN_SEES_PARALLEL` 和 `USE_MPI` 这两个变量。使用它们会被 CMake 静默忽略，导致并行功能不启用。

### MUMPS 产物要求

CMakeLists.txt 第 453 行（非 Windows 平台）期望 MUMPS_DIR 下存在：
```
${MUMPS_DIR}/libdmumps.a
${MUMPS_DIR}/libmumps_common.a
${MUMPS_DIR}/libpord.a
```

### OpenSeesSP 构建条件

- **必须有 MPI**：`OpenSeesSP` 目标定义在 `if(MPI_FOUND)` 块内（第 728 行附近），MPI 找不到则不会生成 OpenSeesSP
- **MUMPS 可选**：不定义 `MUMPS_DIR` 时用 `_NOMUMPS` 宏禁用，但建议启用以获得最佳求解性能
- **ScaLAPACK 必需**：OpenSeesSP 链接列表中包含 `${SCALAPACK_LIBRARIES}`（第 768 行）

### OpenSees 内置的第三方库（无需系统安装）

以下库源码在 `OTHER/` 下，CMake 会自动 `add_subdirectory` 编译，**不需要在系统上安装**：

| 库 | 路径 | 用途 |
|---|---|---|
| SuperLU 5.1.1 | `OTHER/SuperLU_5.1.1` | 直接求解器 |
| UMFPACK | `OTHER/UMFPACK` | 稀疏求解器 |
| AMD | `OTHER/AMD` | 排序算法 |
| ARPACK | `OTHER/ARPACK` | 特征值求解 |
| CSPARSE | `OTHER/CSPARSE` | 稀疏矩阵 |
| ITPACK | `OTHER/ITPACK` | 迭代求解器 |
| Tetgen | `OTHER/tetgen1.4.3` | 网格生成 |
| Triangle | `OTHER/Triangle` | 三角剖分 |
| SuperLU_DIST | `OTHER/SuperLU_DIST_4.3` | 分布式求解（MPI 时启用） |
| METIS | `OTHER/METIS` | 图划分（MPI 时启用） |
| **BLAS** | `OTHER/BLAS` | **LAPACK 找不到时自动编译** |
| **LAPACK** | `OTHER/LAPACK` | **LAPACK 找不到时自动编译** |

> 💡 这意味着即使服务器上没有 BLAS/LAPACK 系统包，OpenSees 也能用内置源码编译。但内置版本性能远低于 OpenBLAS，仅作 fallback。

---

# 阶段一：本地准备（在 WSL 上完成）

> **本阶段目标**：在 WSL 上完成所有不需要服务器权限的工作，产出一个"开箱即用"的部署包，拷到服务器上直接执行。
>
> **预计耗时**：1-2 小时

## 1.1 检查源码架构兼容性

### 1.1.1 检查内联汇编

x86 到 aarch64 迁移最大的风险是内联汇编。执行以下检查：

```bash
cd /home/jumjum/OpenSees

# 检查 SRC/ 下的 C/C++ 源码（用 -E 扩展正则，兼容性更好）
grep -rnE "__asm__|asm\(" SRC/ --include="*.c" --include="*.cpp" --include="*.h" \
  | grep -vE "//|/\*| \*"

# 检查 OTHER/ 下的第三方库
grep -rnE "__asm__|asm\(" OTHER/ --include="*.c" --include="*.cpp" --include="*.h" \
  | grep -vE "//|/\*| \*"
```

> 💡 **如果上面命令仍报错**，用下面这个更保守的版本（分开查两个模式，避免正则转义问题）：
>
> ```bash
> cd /home/jumjum/OpenSees
>
> # 方式二：分两次查，最兼容
> echo "=== 查找 __asm__ ==="
> grep -rn "__asm__" SRC/ OTHER/ --include="*.c" --include="*.cpp" --include="*.h"
>
> echo "=== 查找 asm( ==="
> grep -rn "asm(" SRC/ OTHER/ --include="*.c" --include="*.cpp" --include="*.h"
> ```
>
> 两种方式都应返回空（无匹配），表示源码中无 x86 内联汇编。

**预期结果**：无输出（已验证 OpenSees 源码不含 x86 内联汇编）。

> ✅ **实测确认**（2026-07-18 在 WSL 上执行）：
> - `SRC/` 目录：找到 **0 处** 匹配
> - `OTHER/` 目录：找到 **0 处** 匹配
>
> 结论：OpenSees 源码无 x86 内联汇编，架构兼容性良好。

### 1.1.2 检查 Fortran 源码中的架构相关代码

```bash
# 检查 Fortran 源码（用 -E 扩展正则）
grep -rnE "x86_64|__amd64|__x86_64__" SRC/ OTHER/ \
  --include="*.f" --include="*.F" --include="*.f90" \
  | grep -vE "//|/\*| \*|^.*c "
```

> 💡 **兼容备选**（如果上面报错）：
> ```bash
> grep -rn "x86_64" SRC/ OTHER/ --include="*.f" --include="*.F" --include="*.f90"
> grep -rn "__amd64" SRC/ OTHER/ --include="*.f" --include="*.F" --include="*.f90"
> grep -rn "__x86_64__" SRC/ OTHER/ --include="*.f" --include="*.F" --include="*.f90"
> ```

**预期结果**：无输出。

### 1.1.3 检查 Makefile 中硬编码的 x86 路径

```bash
# 检查 Makefile 模板中的硬编码路径（这些是旧 Makefile 系统，CMake 构建不会用到，但确认一下）
grep -rnE "x86_64|/lib/x86" SRC/ --include="Makefile*" --include="*.incl" --include="*.mak"
```

> 💡 **兼容备选**：
> ```bash
> grep -rn "x86_64" SRC/ --include="Makefile*" --include="*.incl" --include="*.mak"
> grep -rn "/lib/x86" SRC/ --include="Makefile*" --include="*.incl" --include="*.mak"
> ```

**预期结果**：只有 `SRC/system_of_eqn/linearSOE/mumps/Makefile.inc.CARTER` 中有硬编码的 x86 路径，但这是**旧 Makefile 系统的遗留文件，CMake 构建不会使用它**，可忽略。

> ✅ **结论**：OpenSees 源码架构兼容性良好，无需修改任何源码即可在 aarch64 上编译。

---

## 1.2 准备换行符配置

如果源码在 Windows/WSL 之间切换编辑，需要防止 `\r\n` 换行符问题。

### 创建 `.gitattributes` 文件

在 WSL 上执行：

```bash
cd /home/jumjum/OpenSees
cat > .gitattributes << 'EOF'
# 统一使用 LF 换行符，避免 Windows/鲲鹏之间传输时的 ^M 问题
*.cpp   text eol=lf
*.h     text eol=lf
*.c     text eol=lf
*.f     text eol=lf
*.F     text eol=lf
*.f90   text eol=lf
*.tcl   text eol=lf
*.cmake text eol=lf
*.sh    text eol=lf
*.py    text eol=lf
*.md    text eol=lf
EOF

git add .gitattributes
git commit -m "add .gitattributes for cross-platform line endings"
```

> ⚠️ Fortran `.F` 文件尤其重要 —— `\r\n` 换行会导致 MUMPS 的预处理宏被 cpp 错误解析。

---

## 1.3 准备 TCL 源码（关键！）

### 为什么需要准备 TCL 源码？

经查证，**conda-forge 的 aarch64 频道没有 `tcl` 包**（只有 `tk`），而 OpenSees 的 `CMakeLists.txt` 第 126 行有 `find_package(TCL REQUIRED)` —— TCL 是必需依赖。

因此需要准备 TCL 源码，在服务器上用户空间编译。

### 下载 TCL 源码

```bash
cd /home/jumjum
mkdir -p kunpeng_deploy/external_sources
cd kunpeng_deploy/external_sources

# 下载 TCL 8.6 源码（与 WSL 上系统包版本对齐）
wget https://prdownloads.sourceforge.net/tcl/tcl8.6.14-src.tar.gz
tar xzf tcl8.6.14-src.tar.gz
# 验证
ls tcl8.6.14/unix/
```

> 💡 TCL 源码包约 10MB，编译后约 30MB，用户空间编译非常简单（见阶段二）。

---

## 1.4 将 MUMPS 源码合并到 OpenSees 仓库

MUMPS 不能复用 x86 的编译产物（`.a` 文件是 x86 二进制），必须在鲲鹏上重新编译。为简化仓库管理，**将 MUMPS 源码作为子目录合并到 OpenSees 仓库中**，只需维护一个 GitHub 仓库。

### 1.4.1 确认 MUMPS 源码可用

```bash
cd /home/jumjum/mumps
ls
# 应该有: CMakeLists.txt, cmake/, src/, include/, libseq/, pord/ 等

git log --oneline -3
# 确认有提交历史
```

### 1.4.2 将 MUMPS 源码复制到 OpenSees 仓库

```bash
cd /home/jumjum

# 用 rsync 复制，排除 build 产物和 .git 目录
rsync -a \
  --exclude='build' \
  --exclude='build-*' \
  --exclude='.git' \
  --exclude='*.a' \
  --exclude='*.o' \
  --exclude='*.mod' \
  /home/jumjum/mumps/ \
  /home/jumjum/OpenSees/mumps/

# 验证
ls /home/jumjum/OpenSees/mumps/
# 应看到: CMakeLists.txt, cmake/, src/, include/, libseq/, pord/ 等

du -sh /home/jumjum/OpenSees/mumps/
# 约 292KB（纯源码，无编译产物）
```

### 1.4.3 提交 MUMPS 到 OpenSees 仓库

```bash
cd /home/jumjum/OpenSees

# 暂存 mumps 目录
git add mumps/

# 确认 .gitignore 不会误排除 mumps 内的文件
git status --short mumps/ | head -5

# 提交
git commit -m "add mumps source into OpenSees repo (consolidate repos)"

# 推送（与 1.5 节的 OpenSees 推送一起执行即可）
```

> 💡 **为什么不保留独立 MUMPS 仓库？**
> - 只需维护一个 GitHub 仓库，简化管理
> - 鲲鹏服务器上只需 `git clone` 一次，MUMPS 源码随之而来
> - MUMPS 源码仅 292KB，对 OpenSees 仓库体积影响可忽略
> - 版本绑定：OpenSees 和 MUMPS 的版本组合明确记录在同一个 git 历史中

> ⚠️ **注意**：OpenSees 的 `CMakeLists.txt` 中 `MUMPS_DIR` 指向的是 **MUMPS 的编译产物目录**（`build-kunpeng/`），不是源码目录。源码在 `OpenSees/mumps/`，编译产物会在 `OpenSees/mumps/build-kunpeng/`，两者分开。

---

## 1.5 推送 OpenSees 源码（含 MUMPS）到 GitHub

OpenSees 源码（含合并的 MUMPS 子目录）通过单一 GitHub 仓库传输，鲲鹏服务器上一次 `git clone` 即可获取全部源码。

### 1.5.1 在 GitHub 上创建 OpenSees 仓库

1. 打开 https://github.com/new
2. Owner 选 `Tilyes`
3. Repository name 填：`OpenSees_ARM_version`
4. **不要**勾选 Add README / .gitignore / license（保持空仓库）
5. 选 Private（推荐）
6. 点 **Create repository**

### 1.5.2 推送 OpenSees（含 MUMPS）到你的 GitHub

```bash
cd /home/jumjum/OpenSees

# 添加你的 GitHub 仓库为远程（保留官方 origin 方便以后同步上游更新）
git remote add mygithub https://github.com/Tilyes/OpenSees_ARM_version.git
# 如果报 already exists，用 set-url 更新：
# git remote set-url mygithub https://github.com/Tilyes/OpenSees_ARM_version.git

# 验证远程配置
git remote -v
# origin    https://github.com/OpenSees/OpenSees.git             (官方，保留)
# mygithub  https://github.com/Tilyes/OpenSees_ARM_version.git   (你的)

# 推送（包含 1.4 节合并的 mumps/ 目录）
GIT_SSL_NO_VERIFY=1 git push -u mygithub main
```

> 💡 推送时需要认证：
> - **Username**: `Tilyes`
> - **Password**: 你的 GitHub Personal Access Token（不是密码）
>
> 如果没有 token，去 GitHub → Settings → Developer settings → Personal access tokens → Generate new token → 勾选 `repo` 权限。

### 1.5.3 推送迁移文档（可选）

把迁移指南也推上去，方便鲲鹏服务器上查阅：

```bash
cd /home/jumjum/OpenSees
git add kunpeng_migration_guide.md
git commit -m "add kunpeng migration guide"
GIT_SSL_NO_VERIFY=1 git push mygithub main
```

### 1.5.4 确认 `.gitignore` 排除编译产物

推送前确认不会把 build 产物传上去（这些文件很大且无意义）：

```bash
cd /home/jumjum/OpenSees
cat .gitignore 2>/dev/null | head -20
# 应包含: build/, *.o, *.a, *.out, *.mod 等

# 如果没有 .gitignore，创建一个
cat > .gitignore << 'EOF'
# 编译产物
build/
build-*/
*.o
*.a
*.so
*.out
*.mod

# 可执行文件
OpenSees
OpenSeesSP
OpenSeesMP
OpenSeesPy

# IDE 文件
.vscode/
.idea/
EOF

git add .gitignore
git commit -m "add .gitignore to exclude build artifacts"
GIT_SSL_NO_VERIFY=1 git push mygithub main
```

> ⚠️ 如果之前不小心已经把 build 产物加入了 git 暂存区，用以下命令清理：
> ```bash
> git rm -r --cached build/ build-kunpeng/ 2>/dev/null
> git commit -m "remove build artifacts from tracking"
> git push mygithub main
> ```

---

## 1.6 编写服务器部署脚本

### 1.6.1 环境检查脚本

在 WSL 上创建 `check_env.sh`，用于服务器上确认环境：

```bash
cat > /home/jumjum/kunpeng_deploy/check_env.sh << 'SCRIPT'
#!/bin/bash
# 鲲鹏服务器环境检查脚本
# 用法: bash check_env.sh

echo "=========================================="
echo "  鲲鹏服务器环境检查"
echo "=========================================="

echo ""
echo "[1] 系统架构"
echo "    $(uname -m)  (预期: aarch64)"
echo "    $(cat /etc/os-release | grep PRETTY_NAME)"

echo ""
echo "[2] CPU 信息"
lscpu | grep -E "Model name|CPU\(s\)|Architecture"

echo ""
echo "[3] 内存与磁盘"
free -h | grep Mem
echo "    家目录可用空间: $(df -h $HOME | tail -1 | awk '{print $4}')"

echo ""
echo "[4] 已存在的编译器"
for cmd in gcc g++ gfortran clang flang cmake make; do
    if command -v $cmd &>/dev/null; then
        echo "    ✓ $cmd: $($cmd --version 2>&1 | head -1)"
    else
        echo "    ✗ $cmd: 未安装"
    fi
done

echo ""
echo "[5] 已存在的 MPI"
for cmd in mpirun mpicc mpicxx mpif90; do
    if command -v $cmd &>/dev/null; then
        echo "    ✓ $cmd: $($cmd --version 2>&1 | head -1)"
    else
        echo "    ✗ $cmd: 未安装"
    fi
done

echo ""
echo "[6] 已存在的数学库"
for lib in libopenblas libblas liblapack libscalapack; do
    found=$(find /usr/lib* /usr/local/lib $HOME -name "${lib}*.so*" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "    ✓ $lib: $found"
    else
        echo "    ✗ $lib: 未找到"
    fi
done

echo ""
echo "[7] 已存在的其他依赖"
for cmd in tclsh h5cc python3; do
    if command -v $cmd &>/dev/null; then
        echo "    ✓ $cmd: $($cmd --version 2>&1 | head -1)"
    else
        echo "    ✗ $cmd: 未安装"
    fi
done

for hdr in /usr/include/tcl.h /usr/include/eigen3; do
    if [ -e "$hdr" ]; then
        echo "    ✓ $hdr"
    else
        echo "    ✗ $hdr: 未找到"
    fi
done

echo ""
echo "[8] 网络连通性"
if curl -sL --head --max-time 5 https://repo.anaconda.com >/dev/null 2>&1; then
    echo "    ✓ 可访问 repo.anaconda.com (Miniconda 可下载)"
else
    echo "    ✗ 无法访问 repo.anaconda.com"
fi
if curl -sL --head --max-time 5 https://github.com >/dev/null 2>&1; then
    echo "    ✓ 可访问 github.com"
else
    echo "    ✗ 无法访问 github.com"
fi

echo ""
echo "=========================================="
echo "  检查完成。将以上输出保存，用于调整部署方案"
echo "=========================================="
SCRIPT
chmod +x /home/jumjum/kunpeng_deploy/check_env.sh
```

### 1.6.2 一键部署脚本

在 WSL 上创建 `deploy.sh`，这是在服务器上执行的主脚本：

```bash
cat > /home/jumjum/kunpeng_deploy/deploy.sh << 'SCRIPT'
#!/bin/bash
set -e

#==============================================================
# OpenSeesSP 鲲鹏服务器部署脚本（无 root 权限版）
# 所有组件安装在 $HOME 下，不需要 sudo
# 源码通过 git clone 从 GitHub 获取
#==============================================================

# 配置区 —— 可根据 check_env.sh 的结果修改
INSTALL_PREFIX="$HOME/opensees_env"       # 用户空间安装根目录
OPENSEES_SRC="$HOME/OpenSees"             # OpenSees 源码目录（含 mumps/ 子目录）
MUMPS_SRC="$OPENSEES_SRC/mumps"           # MUMPS 源码目录（合并在 OpenSees 仓库内）
TCL_VERSION="8.6.14"                      # TCL 版本
BUILD_JOBS=$(nproc)                       # 编译并行数

# GitHub 仓库地址（单一仓库，含 OpenSees + MUMPS 源码）
GITHUB_USER="Tilyes"
OPENSEES_REPO="https://github.com/${GITHUB_USER}/OpenSees_ARM_version.git"

echo "=========================================="
echo "  OpenSeesSP 鲲鹏部署（用户空间）"
echo "  安装前缀: $INSTALL_PREFIX"
echo "  编译并行: $BUILD_JOBS"
echo "  GitHub 仓库: $OPENSEES_REPO"
echo "=========================================="

mkdir -p "$INSTALL_PREFIX"

#==============================================================
# 步骤 1: 安装 Miniconda（用户空间 Python + 包管理）
#==============================================================
echo ""
echo "[1/7] 安装 Miniconda..."

CONDA_PREFIX="$HOME/miniconda3"
if [ -d "$CONDA_PREFIX" ]; then
    echo "    Miniconda 已存在，跳过"
else
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_PREFIX"
    rm miniconda.sh
    echo "    Miniconda 安装完成: $CONDA_PREFIX"
fi

# 激活 conda
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda init bash 2>/dev/null || true

#==============================================================
# 步骤 2: 用 conda 安装编译工具链和依赖库
#==============================================================
echo ""
echo "[2/7] 安装编译工具链和依赖库（通过 conda-forge）..."

# 创建专用环境，避免污染 base
conda create -y -n opensees python=3.10 2>/dev/null || true
conda activate opensees

# 安装编译器套件（aarch64 交叉编译器）
conda install -y -c conda-forge \
    gcc_linux-aarch64=10 \
    gxx_linux-aarch64=10 \
    gfortran_linux-aarch64=10 \
    cmake \
    make \
    pkg-config \
    git

# 安装运行时依赖库
conda install -y -c conda-forge \
    openmpi \
    openblas \
    scalapack \
    hdf5 \
    eigen \
    zlib

echo "    依赖安装完成"
echo "    conda 环境路径: $CONDA_PREFIX/envs/opensees"

# 记录关键库路径（供后续 CMake 使用）
CONDA_ENV="$CONDA_PREFIX/envs/opensees"
LIB_DIR="$CONDA_ENV/lib"
INC_DIR="$CONDA_ENV/include"

echo "    库目录: $LIB_DIR"
echo "    头文件目录: $INC_DIR"

#==============================================================
# 步骤 3: git clone 源码仓库（单一仓库，含 OpenSees + MUMPS）
#==============================================================
echo ""
echo "[3/7] 从 GitHub 克隆源码（含 MUMPS 子目录）..."

if [ -d "$OPENSEES_SRC" ]; then
    echo "    OpenSees 目录已存在，尝试更新..."
    cd "$OPENSEES_SRC"
    git pull 2>/dev/null || echo "    (git pull 跳过，可能有本地修改)"
else
    echo "    克隆 OpenSees（含 mumps/ 子目录）..."
    git clone "$OPENSEES_REPO" "$OPENSEES_SRC"
    echo "    OpenSees 克隆完成: $OPENSEES_SRC"
fi

# 验证 mumps 子目录存在
if [ ! -d "$MUMPS_SRC" ]; then
    echo "    ✗ 错误: $MUMPS_SRC 不存在，仓库可能不完整"
    echo "      请确认 GitHub 仓库中包含 mumps/ 子目录"
    exit 1
else
    echo "    ✓ MUMPS 源码确认存在: $MUMPS_SRC"
fi

#==============================================================
# 步骤 4: 编译安装 TCL（conda-forge aarch64 无 tcl 包）
#==============================================================
echo ""
echo "[4/7] 编译安装 TCL $TCL_VERSION..."

TCL_PREFIX="$INSTALL_PREFIX/tcl"
if [ -f "$TCL_PREFIX/lib/libtcl8.6.so" ]; then
    echo "    TCL 已编译，跳过"
else
    cd /tmp
    if [ ! -f "tcl${TCL_VERSION}-src.tar.gz" ]; then
        wget -q https://prdownloads.sourceforge.net/tcl/tcl${TCL_VERSION}-src.tar.gz
    fi
    tar xzf tcl${TCL_VERSION}-src.tar.gz
    cd tcl${TCL_VERSION}/unix

    # 用 conda 的编译器
    ./configure \
        --prefix="$TCL_PREFIX" \
        CC="$CONDA_ENV/bin/aarch64-conda-linux-gnu-gcc" \
        CFLAGS="-O2 -fPIC"

    make -j $BUILD_JOBS
    make install
    echo "    TCL 安装完成: $TCL_PREFIX"
fi

#==============================================================
# 步骤 5: 编译 MUMPS（aarch64 原生，源码在 OpenSees/mumps/）
#==============================================================
echo ""
echo "[5/7] 编译 MUMPS（源码: $MUMPS_SRC）..."

MUMPS_BUILD="$MUMPS_SRC/build-kunpeng"
if [ -f "$MUMPS_BUILD/libdmumps.a" ]; then
    echo "    MUMPS 已编译，跳过"
else
    cd "$MUMPS_SRC"
    mkdir -p build-kunpeng && cd build-kunpeng

    # 设置编译器为 conda 的 aarch64 编译器
    export CC="$CONDA_ENV/bin/aarch64-conda-linux-gnu-gcc"
    export CXX="$CONDA_ENV/bin/aarch64-conda-linux-gnu-g++"
    export FC="$CONDA_ENV/bin/aarch64-conda-linux-gnu-gfortran"

    # ScaLAPACK 库链（conda 的 scalapack + openblas）
    SCALAPACK_LIBS="$LIB_DIR/libscalapack.so;$LIB_DIR/libopenblas.so"

    cmake .. -Darith=d \
        -DBLAS_LIBRARIES="$LIB_DIR/libopenblas.so" \
        -DLAPACK_LIBRARIES="$LIB_DIR/libopenblas.so" \
        -DSCALAPACK_LIBRARIES="$SCALAPACK_LIBS"

    cmake --build . --config Release --parallel $BUILD_JOBS
    echo "    MUMPS 编译完成: $MUMPS_BUILD"
    ls -la libdmumps.a libmumps_common.a libpord.a
fi

#==============================================================
# 步骤 6: 编译 OpenSeesSP
#==============================================================
echo ""
echo "[6/7] 编译 OpenSeesSP..."

OPENSEES_BUILD="$OPENSEES_SRC/build-kunpeng"
if [ -f "$OPENSEES_BUILD/OpenSeesSP" ]; then
    echo "    OpenSeesSP 已编译，跳过"
else
    cd "$OPENSEES_SRC"
    mkdir -p build-kunpeng && cd build-kunpeng

    # 设置编译器
    export CC="$CONDA_ENV/bin/aarch64-conda-linux-gnu-gcc"
    export CXX="$CONDA_ENV/bin/aarch64-conda-linux-gnu-g++"
    export FC="$CONDA_ENV/bin/aarch64-conda-linux-gnu-gfortran"

    # ScaLAPACK 库链
    SCALAPACK_LIBS="$LIB_DIR/libscalapack.so;$LIB_DIR/libopenblas.so"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMUMPS_DIR="$MUMPS_BUILD" \
        -DOPENMPI=TRUE \
        -DSCALAPACK_LIBRARIES="$SCALAPACK_LIBS" \
        -DTCL_INCLUDE_PATH="$TCL_PREFIX/include" \
        -DTCL_LIBRARY="$TCL_PREFIX/lib/libtcl8.6.so" \
        -DTCL_TCLSH="$TCL_PREFIX/bin/tclsh8.6" \
        -DHDF5_ROOT="$CONDA_ENV" \
        -DEigen3_DIR="$CONDA_ENV/share/eigen3/cmake"

    cmake --build . --config Release --target OpenSeesSP --parallel $BUILD_JOBS
    echo "    OpenSeesSP 编译完成: $OPENSEES_BUILD/OpenSeesSP"
    file OpenSeesSP
fi

#==============================================================
# 步骤 7: 运行验证
#==============================================================
echo ""
echo "[7/7] 运行验证..."

# 设置运行时库搜索路径
export LD_LIBRARY_PATH="$TCL_PREFIX/lib:$LIB_DIR:$LD_LIBRARY_PATH"

cd "$OPENSEES_BUILD"
echo "    单进程测试..."
./OpenSeesSP ../test2.tcl && echo "    ✓ 单进程测试通过" || echo "    ✗ 单进程测试失败"

echo ""
echo "    多进程测试 (mpirun -np 4)..."
mpirun --oversubscribe -np 4 ./OpenSeesSP ../test2.tcl && echo "    ✓ 多进程测试通过" || echo "    ✗ 多进程测试失败"

#==============================================================
# 输出环境变量总结
#==============================================================
echo ""
echo "=========================================="
echo "  部署完成！"
echo "=========================================="
echo ""
echo "后续使用前，请执行以下命令激活环境："
echo ""
echo "  source $CONDA_PREFIX/etc/profile.d/conda.sh"
echo "  conda activate opensees"
echo "  export LD_LIBRARY_PATH=$TCL_PREFIX/lib:$LIB_DIR:\$LD_LIBRARY_PATH"
echo ""
echo "OpenSeesSP 路径: $OPENSEES_BUILD/OpenSeesSP"
echo "运行示例: mpirun --oversubscribe -np 4 $OPENSEES_BUILD/OpenSeesSP your_script.tcl"
echo ""
echo "更新源码（OpenSees + MUMPS 一起更新）: cd $OPENSEES_SRC && git pull"
SCRIPT
chmod +x /home/jumjum/kunpeng_deploy/deploy.sh
```

> ⚠️ **注意**：
> 1. `deploy.sh` 中的编译器路径 `$CONDA_ENV/bin/aarch64-conda-linux-gnu-gcc` 是 conda-forge 交叉编译器的命名约定。实际名称可能因 conda 版本略有不同，部署时用 `ls $CONDA_ENV/bin/*gcc*` 确认。
> 2. **私有仓库 clone 需要认证**：第一次 `git clone` 会要求输入 GitHub 用户名和 Personal Access Token。如果服务器无法交互式输入，可以提前在 WSL 上生成 token 并嵌入 URL（不推荐长期使用）：
>    ```bash
>    # 临时方式（token 会暴露在 shell 历史，用完即改）
>    git clone https://Tilyes:你的TOKEN@github.com/Tilyes/OpenSees_ARM_version.git
>    ```
>    或配置 git credential helper 缓存凭证：
>    ```bash
>    git config --global credential.helper 'cache --timeout=3600'
>    ```

---

## 1.7 整理部署包

### 最终的部署包结构

源码已推送到 GitHub，部署包只需包含脚本和 TCL 源码（体积很小）：

```bash
cd /home/jumjum
ls -la kunpeng_deploy/
```

预期内容：

```
kunpeng_deploy/
├── check_env.sh                    # 环境检查脚本
├── deploy.sh                       # 一键部署脚本（含 git clone 步骤）
└── external_sources/
    └── tcl8.6.14-src.tar.gz        # TCL 源码（conda-forge aarch64 无此包）
```

> 💡 对比之前的方案：不再需要 `OpenSees-src.tar.gz` 和 `mumps-src.tar.gz`（约 100MB+），因为源码通过 GitHub 传输。部署包仅约 10MB。

### 打包为单一文件便于传输

```bash
cd /home/jumjum
tar czf kunpeng_deploy.tar.gz kunpeng_deploy/
du -sh kunpeng_deploy.tar.gz
# 预期 ~10MB（只有脚本 + TCL 源码）
```

---

## 1.8 本地准备阶段总结

| 工作项 | 状态 | 产物 |
|---|---|---|
| 架构兼容性检查 | ✅ 无内联汇编，无需改源码 | 检查记录 |
| 换行符配置 | ✅ `.gitattributes` | git commit |
| TCL 源码准备 | ✅ 下载 tcl8.6.14 源码 | `tcl8.6.14-src.tar.gz` |
| MUMPS 源码合并到 OpenSees | ✅ 复制到 `OpenSees/mumps/` | git commit |
| OpenSees 源码推送 GitHub | ✅ 推送到 `Tilyes/OpenSees_ARM_version`（含 mumps/） | GitHub 仓库 |
| 环境检查脚本 | ✅ | `check_env.sh` |
| 一键部署脚本 | ✅ 含 git clone 步骤 | `deploy.sh` |
| 统一打包 | ✅ | `kunpeng_deploy.tar.gz` (~10MB) |

> 🎯 **本阶段产出**：
> - 一个 GitHub 仓库 `Tilyes/OpenSees_ARM_version`（含 OpenSees + MUMPS 源码）
> - 一个 `kunpeng_deploy.tar.gz` 文件（脚本 + TCL 源码，约 10MB）
>
> 鲲鹏服务器上解压后，先跑 `check_env.sh` 确认环境，再跑 `deploy.sh` —— 脚本会自动从 GitHub clone 源码（含 MUMPS 子目录）并完成编译。

---

# 阶段二：服务器有限权限部署

> **本阶段目标**：在鲲鹏服务器上，仅用用户权限（`$HOME` 目录内）完成 OpenSeesSP 的编译和运行。
>
> **核心约束**：不使用 `sudo` / `dnf install`，所有工具和库安装在用户目录下。
>
> **预计耗时**：2-4 小时（取决于服务器性能和网络）

## 前置条件

- 鲲鹏服务器账号，能 SSH 登录
- 家目录有至少 5GB 可用空间（Miniconda ~500MB + 源码 ~200MB + 编译产物 ~2GB）
- 能访问外网：
  - 能访问 `repo.anaconda.com`（下载 Miniconda + conda 包）
  - 能访问 `github.com`（clone 源码仓库）
  - 如果 GitHub 访问受限，见下方"情况 C"的离线方案

---

## 2.1 传输部署包到服务器

部署包现在很小（约 10MB，只有脚本 + TCL 源码），源码通过 GitHub 直接 clone，不需要打包传输。

### 从 WSL 上传部署包

```bash
# 在 WSL 上执行
scp /home/jumjum/kunpeng_deploy.tar.gz <用户名>@<鲲鹏服务器IP>:~/
```

### 在服务器上解压

```bash
# SSH 登录鲲鹏服务器后
cd ~
tar xzf kunpeng_deploy.tar.gz
ls kunpeng_deploy/
# 应看到: check_env.sh deploy.sh external_sources/
```

> 💡 如果服务器能直接访问 GitHub，也可以不传部署包，手动创建脚本。但传过去更方便（脚本已经写好）。

---

## 2.2 环境检查

```bash
cd ~/kunpeng_deploy
bash check_env.sh
```

### 根据检查结果调整策略

**情况 A：服务器已有 GCC + MPI + 数学库（系统安装）**

如果 `check_env.sh` 显示 `gcc`、`gfortran`、`mpirun`、`libopenblas`、`libscalapack` 都已存在，说明管理员已预装。此时可以**跳过 Miniconda，直接用系统工具链**：

```bash
# 修改 deploy.sh，注释掉步骤 1-2，改用系统库
# 或直接手动执行步骤 3-7，库路径改为系统路径
```

**情况 B：服务器缺少部分依赖（最常见）**

按 `deploy.sh` 脚本执行，用 Miniconda 补齐缺失依赖。脚本会自动从 GitHub clone 源码。

**情况 C：服务器无法访问 GitHub（但能访问其他外网）**

如果服务器能访问外网但 GitHub 受限，需要在 WSL 上额外准备源码包一起传输：

```bash
# 在 WSL 上：从 GitHub clone 干净的源码（含 mumps/ 子目录），打包
cd /tmp
git clone https://github.com/Tilyes/OpenSees_ARM_version.git

tar czf /home/jumjum/kunpeng_deploy/OpenSees-src.tar.gz \
  --exclude='OpenSees_ARM_version/.git' OpenSees_ARM_version/

# 重新打包部署包
cd /home/jumjum
tar czf kunpeng_deploy.tar.gz kunpeng_deploy/
scp kunpeng_deploy.tar.gz <用户名>@<服务器IP>:~/
```

然后在服务器上修改 `deploy.sh` 步骤 3，把 `git clone` 改为 `tar xzf` 解压本地源码包（源码中已含 `mumps/` 子目录，无需单独处理）。

**情况 D：服务器完全无法访问外网**

需要从 WSL 上额外下载 Miniconda 安装包 + 源码包一起传输：

```bash
# 在 WSL 上下载所有离线资源
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
  -O /home/jumjum/kunpeng_deploy/external_sources/miniconda.sh

# 加上情况 C 的源码包
# 重新打包
cd /home/jumjum
tar czf kunpeng_deploy.tar.gz kunpeng_deploy/
scp kunpeng_deploy.tar.gz <用户名>@<服务器IP>:~/
```

然后在服务器上修改 `deploy.sh`：
- 步骤 1：改为从本地 `miniconda.sh` 安装
- 步骤 3：改为从本地 `tar.gz` 解压源码

---

## 2.3 执行一键部署

```bash
cd ~/kunpeng_deploy
bash deploy.sh
```

### 脚本执行的 7 个步骤详解

#### 步骤 1：安装 Miniconda（用户空间）

- 安装到 `$HOME/miniconda3/`
- 不需要 root，所有文件在用户目录
- 提供后续 `conda install` 的基础环境

#### 步骤 2：用 conda-forge 安装依赖

通过 conda-forge 的 aarch64 频道安装（已验证可用）：

| 包名 | conda-forge aarch64 | 用途 |
|---|---|---|
| `gcc_linux-aarch64` | ✅ 有 | C 编译器 |
| `gxx_linux-aarch64` | ✅ 有 | C++ 编译器 |
| `gfortran_linux-aarch64` | ✅ 有 | Fortran 编译器 |
| `cmake` | ✅ 有 | 构建系统 |
| `git` | ✅ 有 | 版本控制（clone 源码） |
| `openmpi` | ✅ 有 | MPI 实现 |
| `openblas` | ✅ 有 | BLAS（ARM 优化） |
| `scalapack` | ✅ 有 | 并行 ScaLAPACK |
| `hdf5` | ✅ 有 | HDF5 |
| `eigen` | ✅ 有 | Eigen3（头文件库） |
| `zlib` | ✅ 有 | 压缩库 |
| `tcl` | ❌ **无** | 需源码编译（步骤 4） |

> 💡 所有包都安装在 `$HOME/miniconda3/envs/opensees/` 下，完全用户空间。

#### 步骤 3：从 GitHub 克隆源码

脚本自动执行：
```bash
git clone https://github.com/Tilyes/OpenSees_ARM_version.git ~/OpenSees
```

- **单一仓库**：clone 后 `~/OpenSees/mumps/` 子目录即包含 MUMPS 源码，无需额外 clone
- 如果目录已存在，自动 `git pull` 更新（OpenSees + MUMPS 一起更新）
- 脚本会验证 `mumps/` 子目录存在，缺失则报错退出
- **私有仓库需要认证**：第一次 clone 会要求输入 GitHub 用户名和 Personal Access Token
- 建议提前配置 credential helper 缓存凭证：
  ```bash
  git config --global credential.helper 'cache --timeout=3600'
  ```

#### 步骤 4：编译 TCL（源码编译）

- conda-forge aarch64 没有 tcl 包，所以用准备好的源码编译
- 安装到 `$HOME/opensees_env/tcl/`
- 用 conda 提供的 gcc 编译，确保 ABI 兼容

#### 步骤 5：编译 MUMPS（aarch64 原生）

- 源码位置：`$HOME/OpenSees/mumps/`（合并在 OpenSees 仓库内）
- 编译产物位置：`$HOME/OpenSees/mumps/build-kunpeng/`
- 用 conda 的 gfortran 编译
- BLAS/LAPACK/ScaLAPACK 指向 conda 安装的 openblas + scalapack
- 产物：`libdmumps.a`、`libmumps_common.a`、`libpord.a`

#### 步骤 6：编译 OpenSeesSP

- 用 conda 的 gcc/g++ 编译
- `MUMPS_DIR` 指向步骤 5 的产物
- `SCALAPACK_LIBRARIES` 指向 conda 的 scalapack + openblas
- TCL 路径指向步骤 4 的安装
- HDF5、Eigen3 路径指向 conda 环境

#### 步骤 7：运行验证

- 单进程测试：`./OpenSeesSP test2.tcl`
- 多进程测试：`mpirun --oversubscribe -np 4 ./OpenSeesSP test2.tcl`

---

## 2.4 部署后验证

### 验证可执行文件架构

```bash
file ~/OpenSees/build-kunpeng/OpenSeesSP
# 应显示: ELF 64-bit LSB executable, ARM aarch64
```

### 验证动态库依赖

```bash
export LD_LIBRARY_PATH=$HOME/opensees_env/tcl/lib:$HOME/miniconda3/envs/opensees/lib:$LD_LIBRARY_PATH
ldd ~/OpenSees/build-kunpeng/OpenSeesSP | grep -iE "not found|blas|lapack|mpi|mumps|hdf5|tcl"
```

**预期**：所有库都能找到，无 `not found`。

### 验证运行结果

```bash
cd ~/OpenSees/build-kunpeng

# 单进程
./OpenSeesSP ../test2.tcl
echo "Exit code: $?"

# 多进程
mpirun --oversubscribe -np 4 ./OpenSeesSP ../test2.tcl
echo "Exit code: $?"

# 检查输出文件
ls -la ele_forces.out node_disp.out
```

### 与 WSL 结果对比（可选）

```bash
# 在 WSL 上拷出参考结果
scp <WSL上的 ele_forces.out> ~/OpenSees/build-kunpeng/ele_forces_ref.out

# 在服务器上对比
diff ele_forces.out ele_forces_ref.out
# 应无差异（或仅有浮点精度级别的微小差异）
```

---

## 2.5 日常使用配置

### 创建环境激活脚本

```bash
cat > ~/opensees_env/activate.sh << 'EOF'
#!/bin/bash
# OpenSeesSP 环境激活脚本
# 用法: source ~/opensees_env/activate.sh

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate opensees

export LD_LIBRARY_PATH=$HOME/opensees_env/tcl/lib:$HOME/miniconda3/envs/opensees/lib:$LD_LIBRARY_PATH

export OPENSEES_SP=$HOME/OpenSees/build-kunpeng/OpenSeesSP

echo "OpenSeesSP 环境已激活"
echo "可执行文件: $OPENSEES_SP"
echo "运行示例: mpirun --oversubscribe -np 4 $OPENSEES_SP your_script.tcl"
EOF
```

### 后续使用

```bash
# 每次登录后激活环境
source ~/opensees_env/activate.sh

# 运行 OpenSeesSP
mpirun --oversubscribe -np 4 $OPENSEES_SP your_script.tcl
```

---

## 2.6 常见问题排查

| 问题 | 原因 | 解决 |
|---|---|---|
| `conda: command not found` | conda 未初始化 | `source ~/miniconda3/etc/profile.d/conda.sh` |
| `aarch64-conda-linux-gnu-gcc: command not found` | 编译器路径不对 | `ls ~/miniconda3/envs/opensees/bin/*gcc*` 确认实际名称 |
| `cannot find -lscalapack` | conda scalapack 未装或路径不对 | `conda install -c conda-forge scalapack`，检查 `$LIB_DIR` |
| `cannot find -ltcl8.6` | TCL 编译失败或路径不对 | 确认 `$HOME/opensees_env/tcl/lib/libtcl8.6.so` 存在 |
| `undefined reference to dgemm_` | OpenBLAS 未链接 | 确认 `SCALAPACK_LIBRARIES` 中包含 openblas 的 `.so` |
| `mpirun: command not found` | conda 的 openmpi 未激活 | `conda activate opensees` |
| `mpirun: not enough slots` | 进程数超过核心数 | 加 `--oversubscribe` 参数 |
| `error while loading shared libraries` | 运行时库路径未设 | `source ~/opensees_env/activate.sh` |
| 编译时 OOM (内存不足) | `--parallel` 数过大 | 修改 `deploy.sh` 中 `BUILD_JOBS=4` 或更小 |
| conda 下载慢 | 网络问题 | 配置 conda 镜像源（见下方） |
| `git clone: Repository not found` | 私有仓库认证失败 | 见下方"GitHub 认证配置" |
| `git clone: server certificate verification failed` | SSL 证书问题 | `git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt` 或 `GIT_SSL_NO_VERIFY=1 git clone ...` |
| `fatal: could not read Username` | 交互式输入失败 | 用 credential helper 或 token 嵌入 URL（见下方） |

### GitHub 认证配置（私有仓库 clone）

私有仓库 `git clone` 需要认证。推荐用 credential helper 缓存 token：

```bash
# 方式一：缓存凭证（推荐，1 小时内免重复输入）
git config --global credential.helper 'cache --timeout=3600'
git clone https://github.com/Tilyes/OpenSees_ARM_version.git
# 第一次输入用户名 + token，之后 1 小时内免输入

# 方式二：token 嵌入 URL（不推荐长期使用，token 会暴露在 shell 历史）
git clone https://Tilyes:你的TOKEN@github.com/Tilyes/OpenSees_ARM_version.git

# 方式三：SSH key（最安全，一劳永逸）
# 1. 生成 SSH key
ssh-keygen -t ed25519 -C "你的邮箱"
# 2. 把 ~/.ssh/id_ed25519.pub 内容添加到 GitHub → Settings → SSH keys
# 3. 用 SSH 地址 clone
git clone git@github.com:Tilyes/OpenSees_ARM_version.git
```

> 💡 **Personal Access Token 获取**：GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token → 勾选 `repo` 权限 → 生成（只显示一次，记得保存）

### conda 镜像源配置（国内服务器加速）

```bash
# 配置清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### GitHub 访问加速（国内服务器）

如果服务器访问 GitHub 慢，可以配置 git 代理或使用镜像：

```bash
# 方式一：配置 git 代理（如果有代理服务器）
git config --global http.proxy http://代理地址:端口

# 方式二：使用 GitHub 镜像（如 ghproxy）
git clone https://ghproxy.com/https://github.com/Tilyes/OpenSees_ARM_version.git

# 方式三：在 WSL 上打包源码，scp 传到服务器（见 2.2 节情况 C）
```

---

## 2.7 后续源码更新流程

当你在 WSL 上修改了源码并推送到 GitHub 后，在鲲鹏服务器上更新（OpenSees + MUMPS 一起更新，因为是同一个仓库）：

```bash
# 激活环境
source ~/opensees_env/activate.sh

# 更新源码（一次 pull 同时更新 OpenSees 和 MUMPS）
cd ~/OpenSees
git pull

# 如果 MUMPS 源码有改动，先重编 MUMPS
cd ~/OpenSees/mumps/build-kunpeng
cmake --build . --config Release --parallel $(nproc)

# 重编 OpenSeesSP（只编译改动的部分，CMake 会增量编译）
cd ~/OpenSees/build-kunpeng
cmake --build . --config Release --target OpenSeesSP --parallel $(nproc)
```

> 💡 建议在 WSL 上修改源码后，先本地编译验证（x86 版），确认无误再 push 到 GitHub，最后在鲲鹏上 pull + 重编。这样避免在服务器上调试编译错误，节省服务器时间。

---

# 附录：两阶段对比总结

| | 阶段一（本地准备） | 阶段二（服务器部署） |
|---|---|---|
| **执行位置** | WSL (x86_64) | 鲲鹏服务器 (aarch64) |
| **需要权限** | 当前用户权限 | 仅用户目录权限（无 sudo） |
| **主要工作** | 架构检查、推送源码到 GitHub、编写部署脚本 | Miniconda 安装、git clone、依赖安装、编译 |
| **产物** | 2 个 GitHub 仓库 + `kunpeng_deploy.tar.gz` (~10MB) | 可运行的 `OpenSeesSP` |
| **预计耗时** | 1-2 小时 | 2-4 小时 |
| **风险** | 低（不涉及编译） | 中（首次编译可能需调路径） |

### 源码传输方式对比

| 方式 | 体积 | 优点 | 缺点 |
|---|---|---|---|
| **GitHub 仓库（本方案）** | 部署包 ~10MB | 版本管理、增量更新、git pull 即可同步 | 需服务器能访问 GitHub |
| tar 打包 + scp | ~100MB | 不依赖 GitHub | 每次更新都要重新打包传输 |
| git bundle + scp | ~50MB | 离线 git 仓库 | 每次更新都要重新 bundle |

> 💡 本方案采用 GitHub 仓库为主，tar 打包作为离线 fallback（见 2.2 节情况 C/D）。

### 依赖安装策略对比

| 依赖项 | 安装方式 | 原因 |
|---|---|---|
| GCC/G++/gfortran | conda-forge `gcc_linux-aarch64` 等 | 用户空间，无需 sudo |
| OpenMPI | conda-forge `openmpi` | 用户空间，与编译器套件配套 |
| OpenBLAS | conda-forge `openblas` | ARM 优化，用户空间 |
| ScaLAPACK | conda-forge `scalapack` | OpenSeesSP 必需 |
| HDF5 | conda-forge `hdf5` | 用户空间 |
| Eigen3 | conda-forge `eigen` | 头文件库，用户空间 |
| **TCL** | **源码编译** | conda-forge aarch64 无此包 |
| **MUMPS** | **git clone + 源码编译** | 需 aarch64 原生编译，不能用 x86 产物 |
| **OpenSees** | **git clone + 源码编译** | 主体程序 |
| SuperLU/AMD/ARPACK 等 | OpenSees 内置源码 | `OTHER/` 目录，CMake 自动编译 |

---

# 附录 B：GitHub 仓库结构

## 你的 GitHub 仓库一览

部署完成后，你在 GitHub 上会有两个仓库：

```
github.com/Tilyes/
├── OpenSees_ARM_version/     # OpenSees 源码
│   ├── SRC/                  # C++/Fortran 源码
│   ├── OTHER/                # 内置第三方库
│   ├── CMakeLists.txt
│   ├── .gitattributes        # 换行符配置
│   ├── .gitignore            # 排除编译产物
│   ├── kunpeng_migration_guide.md  # 本文档
│   └── test2.tcl             # 测试脚本
│
└── Mumps_ARM_version/        # MUMPS 源码
    ├── src/
    ├── include/
    ├── cmake/
    ├── CMakeLists.txt
    └── ...
```

## 后续更新流程

```
WSL (开发)                        鲲鹏 (部署)
────────────                      ────────────
修改源码                           git pull
本地编译验证 (x86)                 重编译 (aarch64)
git push                          运行测试
```

具体命令见文档 2.7 节"后续源码更新流程"。
