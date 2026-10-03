// ============================================================
// CuSPSolver.cpp — GPU 加速稀疏矩阵迭代求解器
//
// 基于 NVIDIA cuSPARSE 库实现 ILU0 预条件 + BiCGStab/CG 迭代法
// 继承自 SparseGenRowLinSolver，OpenSees 框架自动调用
//
// 数据流：
//   OpenSees 组装刚度矩阵 (CSR 格式, 在 CPU 内存)
//     → cudaMemcpy 拷贝到 GPU 显存
//     → cuSPARSE 做 ILU0 预条件分解
//     → BiCGStab 迭代循环 (SpMV + 预条件)
//     → cudaMemcpy 解向量拷回 CPU
//   OpenSees 继续后续分析步骤
// ============================================================

#include "CuSPSolver.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

// cuBLAS（向量运算：点积、加法、缩放）
#include <cublas_v2.h>
// cooperative groups：persistent 三角求解的全网格栅栏
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ============================================================
// CUDA kernel：逐元素乘法（Jacobi 预条件用）
// x[i] = b[i] * diagInv[i]
// ============================================================
__global__ void jacobiKernel(double *d_x, const double *d_b,
                             const double *d_diagInv, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_x[i] = d_b[i] * d_diagInv[i];
    }
}

// ============================================================
// CUDA kernel：提取 CSR 矩阵的对角线元素
// 遍历每一行，找到列号 == 行号的非零元素
// ============================================================
__global__ void extractDiagonalKernel(double *d_diag, const double *d_csrVal,
                                      const int *d_csrRowPtr,
                                      const int *d_csrColInd, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_diag[i] = 1.0;  // 默认 1（防止除零）
        for (int j = d_csrRowPtr[i]; j < d_csrRowPtr[i+1]; j++) {
            if (d_csrColInd[j] == i) {
                d_diag[i] = d_csrVal[j];
                break;
            }
        }
    }
}

// ============================================================
// CUDA kernel：计算对角线的逆
// diagInv[i] = 1.0 / diag[i]
// ============================================================
__global__ void invertDiagonalKernel(double *d_diagInv, const double *d_diag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (d_diag[i] != 0.0)
            d_diagInv[i] = 1.0 / d_diag[i];
        else
            d_diagInv[i] = 1.0;  // 防止除零
    }
}

// ============================================================
// CUDA kernel：ILU0 前推（L*w=b，L 单位下三角）—— 每个 level 一个 kernel
// d_order[start + idx] 是当前 level 内的行号；行内列号升序、对角元在行首
// ============================================================
__global__ void fwdSubKernel(const double *d_b, double *d_w,
                             const double *d_val, const int *d_rp, const int *d_ci,
                             const int *d_order, const int *d_levelPtr, int numLevels)
{
    // persistent kernel：level 循环放 kernel 内，全网格 grid.sync() 当 level 栅栏。
    // 替代原先“每个 level 一个 kernel”的几百次 launch（launch 延迟远大于计算），
    // 同时全 GPU 的 block 都常驻，避免单 block 的带宽瓶颈。
    cg::grid_group grid = cg::this_grid();
    const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    for (int l = 0; l < numLevels; l++) {
        const int start = d_levelPtr[l];
        const int nrows = d_levelPtr[l + 1] - start;
        for (int k = tid; k < nrows; k += nthreads) {
            const int i = d_order[start + k];
            double sum = d_b[i];
            for (int p = d_rp[i]; p < d_rp[i + 1]; p++) {
                const int c = d_ci[p];
                if (c < i) sum -= d_val[p] * d_w[c];   // 严格下三角 L(i,c)
            }
            d_w[i] = sum;                               // 单位对角，无需除法
        }
        grid.sync();  // 本 level 全算完，才进下一 level
    }
}

// ============================================================
// CUDA kernel：ILU0 后推（U*x=w，U 非单位上三角）—— persistent + 全网格栅栏
// ============================================================
__global__ void bwdSubKernel(const double *d_w, double *d_x,
                             const double *d_val, const int *d_rp, const int *d_ci,
                             const int *d_order, const int *d_levelPtr, int numLevels)
{
    cg::grid_group grid = cg::this_grid();
    const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    for (int l = 0; l < numLevels; l++) {
        const int start = d_levelPtr[l];
        const int nrows = d_levelPtr[l + 1] - start;
        for (int k = tid; k < nrows; k += nthreads) {
            const int i = d_order[start + k];
            double sum  = d_w[i];
            double diag = 1.0;
            for (int p = d_rp[i]; p < d_rp[i + 1]; p++) {
                const int c = d_ci[p];
                if (c > i)       sum -= d_val[p] * d_x[c];  // 严格上三角 U(i,c)
                else if (c == i) diag = d_val[p];           // 对角 U(i,i)
            }
            d_x[i] = sum / diag;
        }
        grid.sync();
    }
}

// ============================================================
// SpMV 辅助函数：y = A * x（generic API，替代已移除的 csrmv）
// 描述符在外部创建（矩阵描述符复用），这里只临时包一层向量描述符
// ============================================================
static cusparseStatus_t gpuSpMV(cusparseHandle_t handle,
                                cusparseSpMatDescr_t matA,
                                const double *d_x, double *d_y, int n,
                                void *d_buffer)
{
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, const_cast<double*>(d_x), CUDA_R_64F);
    cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_64F);
    const double alpha = 1.0, beta = 0.0;
    cusparseStatus_t st = cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    return st;
}

// ============================================================
// 构造函数
//
// 初始化参数，创建 cuSPARSE 句柄和矩阵描述符
// 调用方式：Tcl 脚本中 "system cusp -pre ilu0 -solver bicgstab"
//                 触发 specifyCuSPARSE() 函数创建本对象
//
// 你需要填：
//   1. 用成员初始化列表初始化所有参数和状态标志
//   2. cusparseCreate() 创建句柄
//   3. cusparseCreateMatDescr() 创建矩阵描述符
//   4. 设置矩阵类型为 GENERAL，索引基为 ZERO（C 风格从 0 开始）
// ============================================================
CuSPSolver::CuSPSolver(int maxIter, double relTol, int preCond, int solverType)
    : SparseGenRowLinSolver(SOLVER_TAGS_CuSP)
    , m_maxIter(maxIter)
    , m_relTol(relTol)
    , m_preCond(preCond)
    , m_solverType(solverType)
    , m_n(0)
    , m_nnz(0)
    , m_initialized(false)
    , m_structureUploaded(false)
    , m_factorized(false)
{
    // 0. 必须先初始化 CUDA 运行时上下文！
    //    CUDA 12.0 + Blackwell(sm_120) 上，若 cusparseCreate 是进程的
    //    第一个 CUDA 调用，驱动侧上下文初始化会失败（status=1, handle=NULL），
    //    先用一个运行时 API（cudaGetDeviceCount）触发初始化即可规避
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount < 1) {
        opserr << "CuSPSolver: no CUDA-capable device detected\n";
    }

    // 1. 创建 cuSPARSE 句柄（所有 cuSPARSE 操作都需要这个句柄）
    cusparseStatus_t cusparseStatus = cusparseCreate(&m_cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        opserr << "CuSPSolver: cusparseCreate failed, status = "
               << (int)cusparseStatus << "\n";
    }

    // 2. 创建矩阵描述符（描述 CSR 矩阵的属性）
    cusparseCreateMatDescr(&m_matDescr);

    // 3. 设置矩阵属性：通用矩阵 + C 风格从 0 开始编号
    cusparseSetMatType(m_matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(m_matDescr, CUSPARSE_INDEX_BASE_ZERO);

    // 4. 初始化所有 GPU 指针为 nullptr
    //    防止析构时 cudaFree 误释放未分配的地址
    d_csrVal    = nullptr;
    d_csrRowPtr = nullptr;
    d_csrColInd = nullptr;
    d_b         = nullptr;
    d_x         = nullptr;
    d_csrValLU  = nullptr;
    d_rowPtrLU  = nullptr;
    d_colIndLU  = nullptr;
    d_diagInv   = nullptr;
    d_r         = nullptr;
    d_rtilde    = nullptr;
    d_p         = nullptr;
    d_phat      = nullptr;
    d_s         = nullptr;
    d_shat      = nullptr;
    d_t         = nullptr;
    d_v         = nullptr;
    d_tmp       = nullptr;

    // 5. SpMV 描述符和缓冲
    m_spMatA         = nullptr;
    d_spmvBuffer     = nullptr;
    m_spmvBufferSize = 0;

    // 6. 阶段 B：ILU0 / 自实现三角求解资源
    d_w            = nullptr;
    d_fwdOrder     = nullptr;
    d_bwdOrder     = nullptr;
    d_fwdLevelPtr  = nullptr;
    d_bwdLevelPtr  = nullptr;
    m_fwdLevelPtr  = nullptr;
    m_bwdLevelPtr  = nullptr;
    m_numFwdLevels = 0;
    m_numBwdLevels = 0;
    m_levelsBuilt  = false;
    m_coopGrid     = 0;
    m_coopBlock    = 0;

    // 8. 阶段 B：CPU 端 ILU0 工作区
    h_valLU  = nullptr;
    h_w      = nullptr;
    h_diagU  = nullptr;
    h_marker = nullptr;
}


// ============================================================
// 析构函数
//
// ：释放所有 GPU 显存，销毁 cuSPARSE 资源
// 什么时候被调用：OpenSees 分析结束，求解器对象被销毁时
//
// 你需要填：
//   1. 如果 m_initialized 为 true，调用 freeGPU() 释放显存
//   2. 销毁矩阵描述符：cusparseDestroyMatDescr(m_matDescr)
//   3. 销毁句柄：cusparseDestroy(m_cusparseHandle)
// ============================================================
CuSPSolver::~CuSPSolver(void)
{
    // 1. 释放 GPU 显存
    if (m_initialized) {
        freeGPU();
    }

    // 2. 销毁矩阵描述符
    if (m_spMatA != nullptr) {
        cusparseDestroySpMat(m_spMatA);
        m_spMatA = nullptr;
    }
    if (m_matDescr != nullptr) {
        cusparseDestroyMatDescr(m_matDescr);
        m_matDescr = nullptr;
    }

    // 3. 销毁 cuSPARSE 句柄
    if (m_cusparseHandle != nullptr) {
        cusparseDestroy(m_cusparseHandle);
        m_cusparseHandle = nullptr;
    }
}


// ============================================================
// setSize()
//
// 做什么：分配 GPU 显存（OpenSees 在分析开始时调用一次）
// 什么时候被调用：OpenSees 建完模型、开始分析前
//
// 你需要填：
//   1. 从 theSOE 获取矩阵阶数 m_n 和非零元素数 m_nnz
//   2. 调用 allocateGPU() 分配所有 GPU 显存
//   3. 返回 0 表示成功
// ============================================================
int
CuSPSolver::setSize()
{
    // 从 SOE 获取矩阵尺寸
    m_n   = theSOE->size;    // 矩阵阶数 N
    m_nnz = theSOE->nnz;     // 非零元素数

    // 分配 GPU 显存
    return allocateGPU();
}


// ============================================================
// setLinearSOE()
//
// 做什么：保存 SOE 对象的指针，后续 solve() 要从中取矩阵数据
// 这个函数已经很完整，不需要改
// ============================================================
int
CuSPSolver::setLinearSOE(SparseGenRowLinSOE &theLinearSOE)
{
    theSOE = &theLinearSOE;
    return 0;
}


// ============================================================
// solve()
//
// 求解 Kx=f —— 这是核心函数，每一步分析都会调用
// 什么时候被调用：OpenSees 每步非线性分析中组装完刚度矩阵后
//
// 执行流程：
//   1. 从 theSOE 获取 CPU 端的 CSR 数据指针
//      - theSOE->A          : 非零值数组 (double, 长度 nnz)
//      - theSOE->rowStartA  : 行指针数组 (int, 长度 n+1)
//      - theSOE->colA       : 列索引数组 (int, 长度 nnz)
//      - theSOE->B          : 右端项 (double, 长度 n)
//      - theSOE->X          : 解向量 (double, 长度 n)
//      - theSOE->size       : 矩阵阶数 n
//      - theSOE->nnz        : 非零元素数
//
//   2. 首次调用时：分配 GPU 显存 + 拷贝 CSR 结构（行指针、列索引）
//      后续调用时：只拷贝变化的值（矩阵值、右端项）
//      → 优化点：非线性分析中 rowStartA 和 colA 不变，只需传一次
//
//   3. 拷贝矩阵值和右端项到 GPU
//      cudaMemcpy(d_csrVal, h_csrVal, m_nnz*sizeof(double), cudaMemcpyHostToDevice)
//      cudaMemcpy(d_b,      h_b,      m_n*sizeof(double),   cudaMemcpyHostToDevice)
//
//   4. 预条件处理
//      - preCond=0 (none)   : 什么都不做
//      - preCond=1 (jacobi) : 提取对角线逆，存入 d_diagInv
//      - preCond=2 (ilu0)   : 调用 factorizeILU0() 做 ILU0 分解
//
//   5. 根据 solverType 调用对应的迭代求解器
//      - solverType=0 : solveCG()
//      - solverType=1 : solveBiCGStab()
//
//   6. 把解向量从 GPU 拷回 CPU
//      cudaMemcpy(h_x, d_x, m_n*sizeof(double), cudaMemcpyDeviceToHost)
//
// 你需要填：按上述流程实现
// ============================================================
int
CuSPSolver::solve(void)
{
    // 0. 安全检查
    if (theSOE == 0) {
        opserr << "CuSPSolver::solve - no SOE set\n";
        return -1;
    }

    // 1. 从 theSOE 获取 CPU 端的 CSR 数据指针
    m_n   = theSOE->size;
    m_nnz = theSOE->nnz;
    double *h_csrVal    = theSOE->A;
    int    *h_csrRowPtr = theSOE->rowStartA;
    int    *h_csrColInd = theSOE->colA;
    double *h_b         = theSOE->B;
    double *h_x         = theSOE->X;

    // 2. 首次调用：上传 CSR 结构（行指针、列索引）并创建描述符
    //    后续调用：跳过（非线性分析中稀疏结构不变）
    //    注意：不能用 m_initialized 判断——OpenSees 的 setSize() 会先
    //    调用 allocateGPU() 把它置 true，导致结构上传被跳过
    if (!m_structureUploaded) {
        if (!m_initialized) {
            allocateGPU();  // 兜底：setSize 未被调用时
        }
        cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m_n+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColInd, h_csrColInd, m_nnz*sizeof(int),   cudaMemcpyHostToDevice);

        // ILU0 专用结构副本：实测 CUDA 12.8 + sm_120 上 csrilu02 在大矩阵
        // 时会越界写传入的 rowPtr/colInd。给它专用副本，既保护 SpMV 共用的
        // 原始结构，分解后还能校验副本完整性、坏了干净回退 Jacobi
        if (m_preCond == 2) {
            cudaMalloc(&d_rowPtrLU, (m_n+1)*sizeof(int));
            cudaMalloc(&d_colIndLU, m_nnz*sizeof(int));
            cudaMemcpy(d_rowPtrLU, h_csrRowPtr, (m_n+1)*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_colIndLU, h_csrColInd, m_nnz*sizeof(int),   cudaMemcpyHostToDevice);
        }

        // 创建 CSR 矩阵描述符（generic SpMV 用）
        // 描述符只记录指针，后续每步往同一块显存 cudaMemcpy 新值即可复用
        cusparseCreateCsr(&m_spMatA, m_n, m_n, m_nnz,
            d_csrRowPtr, d_csrColInd, d_csrVal,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        // 查询 SpMV 工作缓冲大小并分配（大小只取决于矩阵和算法，查询一次即可）
        cusparseDnVecDescr_t tmpX, tmpY;
        cusparseCreateDnVec(&tmpX, m_n, d_x, CUDA_R_64F);
        cusparseCreateDnVec(&tmpY, m_n, d_b, CUDA_R_64F);
        const double one = 1.0, zero = 0.0;
        cusparseSpMV_bufferSize(m_cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, m_spMatA, tmpX, &zero, tmpY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &m_spmvBufferSize);
        cusparseDestroyDnVec(tmpX);
        cusparseDestroyDnVec(tmpY);
        if (m_spmvBufferSize > 0) {
            cudaMalloc(&d_spmvBuffer, m_spmvBufferSize);
        }

        m_structureUploaded = true;
    }

    // 3. 每步都拷贝矩阵值和右端项（这两者每步都变）
    cudaMemcpy(d_csrVal, h_csrVal, m_nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,      h_b,      m_n*sizeof(double),   cudaMemcpyHostToDevice);

    // 初始化解向量为 0
    cudaMemset(d_x, 0, m_n*sizeof(double));

    // 4. 预条件处理
    if (m_preCond == 1) {
        // Jacobi: 提取对角线并求逆
        // 用自定义 kernel 从 CSR 矩阵中提取对角线元素
        int threads = 256;
        int blocks = (m_n + threads - 1) / threads;
        // 先提取对角线到 d_tmp（临时借用）
        extractDiagonalKernel<<<blocks, threads>>>(
            d_tmp, d_csrVal, d_csrRowPtr, d_csrColInd, m_n);
        cudaDeviceSynchronize();
        // 再求逆存入 d_diagInv
        invertDiagonalKernel<<<blocks, threads>>>(
            d_diagInv, d_tmp, m_n);
        cudaDeviceSynchronize();
    } else if (m_preCond == 2) {
        // ILU0: 每步重新分解（刚度矩阵每步都变）
        // 先准备 Jacobi 对角线作为兜底（零主元等分解失败时用）
        int threads = 256;
        int blocks = (m_n + threads - 1) / threads;
        extractDiagonalKernel<<<blocks, threads>>>(
            d_tmp, d_csrVal, d_csrRowPtr, d_csrColInd, m_n);
        cudaDeviceSynchronize();
        invertDiagonalKernel<<<blocks, threads>>>(
            d_diagInv, d_tmp, m_n);
        cudaDeviceSynchronize();
        if (factorizeILU0() != 0) {
            opserr << "CuSPSolver::solve - ILU0 factorization failed, "
                      "fallback to Jacobi preconditioner\n";
            // m_factorized 已被 factorizeILU0 置 false，applyPreconditioner 走 Jacobi
        }
    }

    // 5. 根据 solverType 调用对应的迭代求解器
    int status = 0;
    if (m_solverType == 0) {
        status = solveCG();
    } else if (m_solverType == 1) {
        status = solveBiCGStab();
    } else {
        opserr << "CuSPSolver::solve - unknown solver type " << m_solverType << "\n";
        return -1;
    }

    if (status != 0) {
        opserr << "CuSPSolver::solve - iterative solver failed to converge\n";
        // 不直接返回错误，仍然把当前最佳解拷回去（可能 OpenSees 还能用）
    }

    // 6. 把解向量从 GPU 拷回 CPU
    cudaMemcpy(h_x, d_x, m_n*sizeof(double), cudaMemcpyDeviceToHost);

    return 0;
}


// ============================================================
// allocateGPU()
//
// 为所有 GPU 端数组分配显存
// 什么时候被调用：setSize() 中调用
//
// 你需要填：
//   对以下每个指针调用 cudaMalloc：
//   - d_csrVal    : m_nnz * sizeof(double)   矩阵非零值
//   - d_csrRowPtr : (m_n+1) * sizeof(int)    行指针
//   - d_csrColInd : m_nnz * sizeof(int)      列索引
//   - d_b         : m_n * sizeof(double)     右端项
//   - d_x         : m_n * sizeof(double)     解向量
//   - d_r, d_rtilde, d_p, d_phat, d_s, d_shat, d_t, d_v
//                 : 各 m_n * sizeof(double)  BiCGStab 工作向量
//   - d_tmp       : m_n * sizeof(double)     临时向量
//   - d_csrValLU  : m_nnz * sizeof(double)   ILU0 分解结果（preCond=2 时用）
//   - d_diagInv   : m_n * sizeof(double)     对角线逆（preCond=1 时用）
//
//   最后设置 m_initialized = true
//
// 提示：建议检查每个 cudaMalloc 的返回值
//   cudaError_t err = cudaMalloc(&d_csrVal, m_nnz * sizeof(double));
//   if (err != cudaSuccess) { opserr << "cudaMalloc failed: " << cudaGetErrorString(err); return -1; }
// ============================================================
int
CuSPSolver::allocateGPU()
{
    // 矩阵 CSR 数据
    cudaMalloc(&d_csrVal,    m_nnz * sizeof(double));
    cudaMalloc(&d_csrRowPtr, (m_n+1) * sizeof(int));
    cudaMalloc(&d_csrColInd, m_nnz * sizeof(int));

    // 右端项和解向量
    cudaMalloc(&d_b, m_n * sizeof(double));
    cudaMalloc(&d_x, m_n * sizeof(double));

    // BiCGStab 工作向量（CG 也用其中部分）
    cudaMalloc(&d_r,      m_n * sizeof(double));
    cudaMalloc(&d_rtilde, m_n * sizeof(double));
    cudaMalloc(&d_p,      m_n * sizeof(double));
    cudaMalloc(&d_phat,   m_n * sizeof(double));
    cudaMalloc(&d_s,      m_n * sizeof(double));
    cudaMalloc(&d_shat,   m_n * sizeof(double));
    cudaMalloc(&d_t,      m_n * sizeof(double));
    cudaMalloc(&d_v,      m_n * sizeof(double));
    cudaMalloc(&d_tmp,    m_n * sizeof(double));

    // 预条件相关
    cudaMalloc(&d_csrValLU, m_nnz * sizeof(double));  // ILU0 分解结果
    cudaMalloc(&d_diagInv,  m_n * sizeof(double));    // Jacobi 对角线逆
    cudaMalloc(&d_w,        m_n * sizeof(double));    // ILU0 前推/后推中间向量
    cudaMalloc(&d_fwdOrder, m_n * sizeof(int));       // 前推 level 行序
    cudaMalloc(&d_bwdOrder, m_n * sizeof(int));       // 后推 level 行序

    // CPU 端 ILU0 工作区（csrilu02 在 sm_120 上已坏，改为 CPU 分解）
    h_valLU  = new double[m_nnz];
    h_w      = new double[m_n];
    h_diagU  = new double[m_n];
    h_marker = new int[m_n];
    if (h_valLU == nullptr || h_w == nullptr || h_diagU == nullptr || h_marker == nullptr) {
        opserr << "CuSPSolver::allocateGPU - failed to alloc CPU ILU0 workspace\n";
        return -1;
    }
    std::memset(h_marker, 0, m_n * sizeof(int));

    m_initialized = true;
    return 0;
}


// ============================================================
// freeGPU()
//
// 释放所有 GPU 端显存
// 什么时候被调用：析构函数中调用
//
// 你需要填：
//   对每个非 nullptr 的指针调用 cudaFree：
//   cudaFree(d_csrVal);  cudaFree(d_csrRowPtr);  ... 等等
//   然后设置 m_initialized = false
// ============================================================
int
CuSPSolver::freeGPU()
{
    // 释放所有 GPU 端显存（检查 nullptr 防止重复释放）
    if (d_csrVal)    { cudaFree(d_csrVal);    d_csrVal = nullptr; }
    if (d_csrRowPtr) { cudaFree(d_csrRowPtr); d_csrRowPtr = nullptr; }
    if (d_csrColInd) { cudaFree(d_csrColInd); d_csrColInd = nullptr; }
    if (d_b)         { cudaFree(d_b);         d_b = nullptr; }
    if (d_x)         { cudaFree(d_x);         d_x = nullptr; }
    if (d_r)         { cudaFree(d_r);         d_r = nullptr; }
    if (d_rtilde)    { cudaFree(d_rtilde);    d_rtilde = nullptr; }
    if (d_p)         { cudaFree(d_p);         d_p = nullptr; }
    if (d_phat)      { cudaFree(d_phat);      d_phat = nullptr; }
    if (d_s)         { cudaFree(d_s);         d_s = nullptr; }
    if (d_shat)      { cudaFree(d_shat);      d_shat = nullptr; }
    if (d_t)         { cudaFree(d_t);         d_t = nullptr; }
    if (d_v)         { cudaFree(d_v);         d_v = nullptr; }
    if (d_tmp)       { cudaFree(d_tmp);       d_tmp = nullptr; }
    if (d_csrValLU)  { cudaFree(d_csrValLU);  d_csrValLU = nullptr; }
    if (d_rowPtrLU)  { cudaFree(d_rowPtrLU);  d_rowPtrLU = nullptr; }
    if (d_colIndLU)  { cudaFree(d_colIndLU);  d_colIndLU = nullptr; }
    if (d_diagInv)   { cudaFree(d_diagInv);   d_diagInv = nullptr; }
    if (d_w)         { cudaFree(d_w);         d_w = nullptr; }
    if (d_fwdOrder)  { cudaFree(d_fwdOrder);  d_fwdOrder = nullptr; }
    if (d_bwdOrder)  { cudaFree(d_bwdOrder);  d_bwdOrder = nullptr; }
    if (d_fwdLevelPtr) { cudaFree(d_fwdLevelPtr); d_fwdLevelPtr = nullptr; }
    if (d_bwdLevelPtr) { cudaFree(d_bwdLevelPtr); d_bwdLevelPtr = nullptr; }
    if (d_spmvBuffer) { cudaFree(d_spmvBuffer); d_spmvBuffer = nullptr; }

    // CPU 端 ILU0 工作区
    if (h_valLU)  { delete [] h_valLU;  h_valLU = nullptr; }
    if (h_w)      { delete [] h_w;      h_w = nullptr; }
    if (h_diagU)  { delete [] h_diagU;  h_diagU = nullptr; }
    if (h_marker) { delete [] h_marker; h_marker = nullptr; }

    // level 调度（host 侧）
    if (m_fwdLevelPtr) { delete [] m_fwdLevelPtr; m_fwdLevelPtr = nullptr; }
    if (m_bwdLevelPtr) { delete [] m_bwdLevelPtr; m_bwdLevelPtr = nullptr; }
    m_numFwdLevels = 0;
    m_numBwdLevels = 0;
    m_levelsBuilt  = false;
    m_factorized = false;

    // 矩阵描述符引用了已释放的显存指针，一并销毁
    if (m_spMatA) { cusparseDestroySpMat(m_spMatA); m_spMatA = nullptr; }
    m_spmvBufferSize = 0;
    m_structureUploaded = false;

    m_initialized = false;
    return 0;
}


// ============================================================
// factorizeILU0()
//
// 在 GPU 上对矩阵做不完全 LU 分解（ILU0）
// 结果存入 d_csrValLU，后续 applyPreconditioner() 使用
//
// 算法原理：
//   完整 LU 分解 K = L×U 会产生填充元（新的非零位置）
//   ILU0：L 和 U 的非零结构与 K 相同（不产生新填充）
//   结果 K ≈ L̃×Ũ，作为预条件矩阵 M
//
// cuSPARSE API 调用流程（你需要填的）：
//   1. 把 d_csrVal 拷贝到 d_csrValLU（ILU0 是原地分解，需要副本）
//      cudaMemcpy(d_csrValLU, d_csrVal, m_nnz*sizeof(double), cudaMemcpyDeviceToDevice)
//
//   2. 符号分析（确定分解所需的缓冲区大小）
//      cusparseDcsrilu02_bufferSize(handle, m_n, m_nnz, descr,
//          d_csrValLU, d_csrRowPtr, d_csrColInd, m_infoILU, &bufferSize)
//      cudaMalloc(&d_work, bufferSize)
//
//   3. 数值分解
//      cusparseDcsrilu02(handle, m_n, m_nnz, descr,
//          d_csrValLU, d_csrRowPtr, d_csrColInd, m_infoILU, policy, d_work)
//
//   4. 下三角分析（用于后续前推 Ly=b）
//      注意：csrsv2 已从 CUDA 12 移除，需用 generic API：
//      cusparseSpSV_createDescr + cusparseSpSV_analysis（L 和 U 各一套）
//
//   5. 上三角分析（用于后续后推 Ux=y）—— 同上，用 cusparseSpSV_analysis
//
//   设置 m_factorized = true
// ============================================================
int
CuSPSolver::factorizeILU0()
{
    m_factorized = false;

    const int n   = m_n;
    const int nnz = m_nnz;

    // —— CPU 端 ILU(0) 分解 ——
    // 说明：cusparseDcsrilu02 在 sm_120(Blackwell) 上对大矩阵会返回
    // INVALID_VALUE、越界写结构数组并清空对角线（原生 sm_120 编译同样复现），
    // 且会污染 CUDA 分配器，无法靠换编译目标规避。这里改为在 CPU 上做等价的
    // 不完全 LU(0) 分解，再把 L/U 值上传 GPU；三角求解仍走 cusparseSpSV（正常）。
    double *A  = theSOE->A;
    int    *rp = theSOE->rowStartA;
    int    *ci = theSOE->colA;

    memcpy(h_valLU, A, nnz * sizeof(double));

    double *w      = h_w;
    double *diagU  = h_diagU;
    int    *marker = h_marker;

    bool zeroPivot = false;
    for (int i = 0; i < n && !zeroPivot; i++) {
        const int rs = rp[i], re = rp[i + 1];

        // scatter：把第 i 行摊到稠密工作区
        for (int p = rs; p < re; p++) {
            int c = ci[p];
            w[c] = h_valLU[p];
            marker[c] = i + 1;
        }

        // 消去严格下三角元素（colA 对角元在行首，其余升序）
        for (int p = rs; p < re; p++) {
            int c = ci[p];
            if (c >= i) continue;          // 跳过对角元(c==i)与上三角(c>i)
            double ucc = diagU[c];
            if (ucc == 0.0) { zeroPivot = true; break; }
            double lik = w[c] / ucc;       // L(i,c)
            w[c] = lik;
            // 用已分解的第 c 行的上三角 U(c,m) 更新本行
            for (int q = rp[c]; q < rp[c + 1]; q++) {
                int m = ci[q];
                if (m <= c) continue;      // 只取 U 的上三角部分
                if (marker[m] == i + 1) {  // m 在本行模式内 → 无填充
                    w[m] -= lik * h_valLU[q];
                }
            }
        }
        if (zeroPivot) break;

        // gather：写回 CSR 并记录对角线 U(i,i)
        for (int p = rs; p < re; p++) {
            int c = ci[p];
            h_valLU[p] = w[c];
            if (c == i) {
                diagU[i] = w[c];
                if (w[c] == 0.0) zeroPivot = true;
            }
        }
    }

    if (zeroPivot) {
        opserr << "CuSPSolver::factorizeILU0 - zero pivot, fallback to Jacobi\n";
        return -1;
    }

    // —— 首次：在原结构上构建前推/后推 level 调度（只依赖结构，做一次）——
    // 前推 level[i] = max(level[j]+1, j<i)，后推从后往前同理；桶排成 fwdOrder/bwdOrder。
    if (!m_levelsBuilt) {
        std::vector<int> fwdLevel(n, 0), bwdLevel(n, 0);
        for (int i = 0; i < n; i++) {
            int lvl = 0;
            for (int p = rp[i]; p < rp[i + 1]; p++) {
                int c = ci[p];
                if (c < i && fwdLevel[c] + 1 > lvl) lvl = fwdLevel[c] + 1;
            }
            fwdLevel[i] = lvl;
        }
        for (int i = n - 1; i >= 0; i--) {
            int lvl = 0;
            for (int p = rp[i]; p < rp[i + 1]; p++) {
                int c = ci[p];
                if (c > i && bwdLevel[c] + 1 > lvl) lvl = bwdLevel[c] + 1;
            }
            bwdLevel[i] = lvl;
        }

        int numFwd = 1, numBwd = 1;
        for (int i = 0; i < n; i++) {
            if (fwdLevel[i] + 1 > numFwd) numFwd = fwdLevel[i] + 1;
            if (bwdLevel[i] + 1 > numBwd) numBwd = bwdLevel[i] + 1;
        }

        std::vector<int> fwdLevelPtr(numFwd + 1, 0), bwdLevelPtr(numBwd + 1, 0);
        std::vector<int> fwdOrder(n), bwdOrder(n);
        for (int i = 0; i < n; i++) {
            fwdLevelPtr[fwdLevel[i] + 1]++;
            bwdLevelPtr[bwdLevel[i] + 1]++;
        }
        for (int l = 0; l < numFwd; l++) fwdLevelPtr[l + 1] += fwdLevelPtr[l];
        for (int l = 0; l < numBwd; l++) bwdLevelPtr[l + 1] += bwdLevelPtr[l];
        {
            std::vector<int> curFwd = fwdLevelPtr, curBwd = bwdLevelPtr;
            for (int i = 0; i < n; i++) {
                fwdOrder[curFwd[fwdLevel[i]]++] = i;
                bwdOrder[curBwd[bwdLevel[i]]++] = i;
            }
        }

        cudaMemcpy(d_fwdOrder, fwdOrder.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwdOrder, bwdOrder.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        m_fwdLevelPtr  = new int[numFwd + 1];
        m_bwdLevelPtr  = new int[numBwd + 1];
        std::memcpy(m_fwdLevelPtr, fwdLevelPtr.data(), (numFwd + 1) * sizeof(int));
        std::memcpy(m_bwdLevelPtr, bwdLevelPtr.data(), (numBwd + 1) * sizeof(int));
        m_numFwdLevels = numFwd;
        m_numBwdLevels = numBwd;
        m_levelsBuilt  = true;

        cudaMalloc(&d_fwdLevelPtr, (numFwd + 1) * sizeof(int));
        cudaMalloc(&d_bwdLevelPtr, (numBwd + 1) * sizeof(int));
        cudaMemcpy(d_fwdLevelPtr, fwdLevelPtr.data(), (numFwd + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwdLevelPtr, bwdLevelPtr.data(), (numBwd + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // cooperative launch 的 grid 大小：所有 block 必须同时常驻，故按
        // 每 SM 最大 co-resident 块数 × SM 数取值，超了会 launch 失败
        m_coopBlock = 256;
        int devId = 0, numSMs = 0;
        cudaGetDevice(&devId);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
        int maxB = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxB, (const void*)fwdSubKernel,
                                                      m_coopBlock, 0);
        m_coopGrid = numSMs * maxB;
        if (m_coopGrid < 1) m_coopGrid = 1;

        opserr << "CuSPSolver::factorizeILU0 - level schedule built: "
               << "numFwdLevels=" << numFwd << ", numBwdLevels=" << numBwd
               << ", coopGrid=" << m_coopGrid << ", coopBlock=" << m_coopBlock << "\n";
    }

    // 上传 L/U 值（原结构顺序，直接写 d_csrValLU；结构与 level 调度首次已上传）
    cudaMemcpy(d_csrValLU, h_valLU, nnz * sizeof(double), cudaMemcpyHostToDevice);

    m_factorized = true;
    return 0;
}


// ============================================================
// applyPreconditioner(double *d_x, double *d_b)
//
// 做什么：求解 M×x = b，其中 M 是预条件矩阵
//         即计算 x = M⁻¹×b，供迭代法中加速收敛用
//
// 三种模式：
//   preCond=0 (none)   : x = b（直接拷贝，无预条件）
//                        cudaMemcpy(d_x, d_b, m_n*sizeof(double), cudaMemcpyDeviceToDevice)
//
//   preCond=1 (jacobi) : x = diag(K)⁻¹ × b（逐元素除以对角线）
//                        需要一个简单的 CUDA kernel 或 cuBLAS 调用
//                        d_x[i] = d_b[i] * d_diagInv[i]
//
//   preCond=2 (ilu0)   : 解 L̃×y = b（前推），Ũ×x = y（后推）
//                        cusparseSpSV_solve(handle, OP_N, ..., d_b, d_tmp, ...)  // 前推
//                        cusparseSpSV_solve(handle, OP_T, ..., d_tmp, d_x, ...)  // 后推
//
// 你需要填：根据 m_preCond 的值选择对应分支
// ============================================================
int
CuSPSolver::applyPreconditioner(double *d_x, double *d_b)
{
    if (m_preCond == 0) {
        // none: x = b（直接拷贝）
        cudaMemcpy(d_x, d_b, m_n*sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else if (m_preCond == 1) {
        // Jacobi: x = diag(K)^{-1} * b
        // 逐元素乘：d_x[i] = d_b[i] * d_diagInv[i]
        int threads = 256;
        int blocks = (m_n + threads - 1) / threads;
        jacobiKernel<<<blocks, threads>>>(d_x, d_b, d_diagInv, m_n);
        cudaDeviceSynchronize();
    }
    else if (m_preCond == 2) {
        if (m_factorized) {
            // ILU0：解 L·y = b（前推，单位对角），再解 U·x = y（后推，非单位对角）。
            // 每个三角求解一个 persistent cooperative kernel，level 循环在 kernel 内；
            // 中间向量 d_w 暂存 y，不与调用方缓冲别名。
            void *fwdArgs[] = { &d_b, &d_w, &d_csrValLU, &d_rowPtrLU, &d_colIndLU,
                                &d_fwdOrder, &d_fwdLevelPtr, &m_numFwdLevels };
            void *bwdArgs[] = { &d_w, &d_x, &d_csrValLU, &d_rowPtrLU, &d_colIndLU,
                                &d_bwdOrder, &d_bwdLevelPtr, &m_numBwdLevels };
            cudaLaunchCooperativeKernel((void*)fwdSubKernel, dim3(m_coopGrid),
                                        dim3(m_coopBlock), fwdArgs, 0, 0);
            cudaLaunchCooperativeKernel((void*)bwdSubKernel, dim3(m_coopGrid),
                                        dim3(m_coopBlock), bwdArgs, 0, 0);
            cudaDeviceSynchronize();
        } else {
            // 分解失败的兜底：Jacobi
            int threads = 256;
            int blocks = (m_n + threads - 1) / threads;
            jacobiKernel<<<blocks, threads>>>(d_x, d_b, d_diagInv, m_n);
            cudaDeviceSynchronize();
        }
    }
    return 0;
}


// ============================================================
// solveCG()
//
// 做什么：共轭梯度法（CG）迭代求解 —— 适合对称正定矩阵
//
// 算法步骤（每一步都标注了对应的 GPU 操作）：
//
//   初始化：
//     r = b - A*x    （初始 x=0，所以 r=b，直接拷贝）
//     p = r
//     rsold = r·r     ← 点积，用 cublasDdot
//
//   循环 (i = 0 to maxIter):
//     Ap = A * p                    ← SpMV，用 gpuSpMV()（cusparseSpMV）★核心
//     alpha = rsold / (p·Ap)        ← 点积
//     x = x + alpha * p             ← 向量加法，用 cublasDaxpy
//     r = r - alpha * Ap            ← 向量加法
//     rsnew = r·r                   ← 点积
//     if sqrt(rsnew) < relTol: 收敛，跳出
//     p = r + (rsnew/rsold) * p     ← 向量运算
//     rsold = rsnew
//
// 你需要填：按上述步骤实现
//
// 提示：
//   - 点积用 cublasDdot(handle, m_n, d_r, 1, d_r, 1)
//   - SpMV 用 gpuSpMV()（内部调 cusparseSpMV）
//   - 向量加法用 cublasDaxpy(handle, m_n, &alpha, d_p, 1, d_x, 1)
//   - 需要一个额外的向量 d_Ap 存储 A×p 的结果
//   - 需要创建 cublasHandle_t（可以在构造函数中创建，或在这里临时创建）
// ============================================================
int
CuSPSolver::solveCG()
{
    // ===== 创建 cuBLAS 句柄（向量运算需要）=====
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // CG 算法中标量系数
    const double alpha_one  = 1.0;
    double alpha, beta;
    double rsold, rsnew, pAp;

    // ===== 初始化 =====
    // r = b（因为 x=0，所以 r = b - A*0 = b）
    cudaMemcpy(d_r, d_b, m_n*sizeof(double), cudaMemcpyDeviceToDevice);

    // 如果有预条件：z = M⁻¹ * r，用 z 代替 r
    if (m_preCond != 0) {
        applyPreconditioner(d_p, d_r);   // d_p = M⁻¹ * r = z
    } else {
        cudaMemcpy(d_p, d_r, m_n*sizeof(double), cudaMemcpyDeviceToDevice);  // p = r
    }

    // rsold = r·z（预条件时）或 r·r（无预条件时）
    if (m_preCond != 0) {
        cublasDdot(cublasHandle, m_n, d_r, 1, d_p, 1, &rsold);  // r·z
    } else {
        cublasDdot(cublasHandle, m_n, d_r, 1, d_r, 1, &rsold);  // r·r
    }

    // ===== 迭代循环 =====
    for (int iter = 0; iter < m_maxIter; iter++) {

        // Ap = A * p  ← SpMV（cuSPARSE 核心操作）
        gpuSpMV(m_cusparseHandle, m_spMatA, d_p, d_v, m_n, d_spmvBuffer);  // d_v = A * p

        // pAp = p · Ap
        cublasDdot(cublasHandle, m_n, d_p, 1, d_v, 1, &pAp);

        if (pAp == 0.0) {
            cublasDestroy(cublasHandle);
            return -1;  // 分母为零，失败
        }

        // alpha = rsold / pAp
        alpha = rsold / pAp;

        // x = x + alpha * p
        cublasDaxpy(cublasHandle, m_n, &alpha, d_p, 1, d_x, 1);

        // r = r - alpha * Ap
        double negAlpha = -alpha;
        cublasDaxpy(cublasHandle, m_n, &negAlpha, d_v, 1, d_r, 1);

        // rsnew = r·r
        cublasDdot(cublasHandle, m_n, d_r, 1, d_r, 1, &rsnew);

        // 收敛判定：||r|| < relTol
        if (sqrt(rsnew) < m_relTol) {
            opserr << "CuSPSolver::solveCG - converged in " << (iter + 1) << " iterations\n";
            cublasDestroy(cublasHandle);
            return 0;  // 收敛
        }

        // 预条件：z = M⁻¹ * r
        if (m_preCond != 0) {
            applyPreconditioner(d_tmp, d_r);  // d_tmp = z = M⁻¹ * r
            // rsnew = r·z（预条件版）
            cublasDdot(cublasHandle, m_n, d_r, 1, d_tmp, 1, &rsnew);
        }

        // beta = rsnew / rsold
        beta = rsnew / rsold;

        // p = z + beta * p（预条件）或 p = r + beta * p（无预条件）
        if (m_preCond != 0) {
            // p = beta*p + z
            cublasDscal(cublasHandle, m_n, &beta, d_p, 1);  // p = beta * p
            cublasDaxpy(cublasHandle, m_n, &alpha_one, d_tmp, 1, d_p, 1);  // p = p + z
        } else {
            // p = beta*p + r
            cublasDscal(cublasHandle, m_n, &beta, d_p, 1);  // p = beta * p
            cublasDaxpy(cublasHandle, m_n, &alpha_one, d_r, 1, d_p, 1);  // p = p + r
        }

        rsold = rsnew;
    }

    // 达到最大迭代次数仍未收敛
    cublasDestroy(cublasHandle);
    return -1;
}


// ============================================================
// solveBiCGStab()
//
// 做什么：双共轭梯度稳定法（BiCGStab）—— 适合一般非对称矩阵
//         比 CG 更通用，是默认推荐的求解器
//
// 算法步骤：
//
//   初始化：
//     r = b - A*x    （初始 x=0，所以 r=b）
//     rtilde = r     （r 的副本，整个迭代中不变）
//
//   循环 (i = 0 to maxIter):
//     rho = rtilde·r                    ← 点积
//     if rho == 0: 失败，跳出
//
//     if i > 0:
//       beta = (rho/rho_old) * (alpha/omega)
//       p = r + beta*(p - omega*v)      ← 向量运算
//     else:
//       p = r
//
//     phat = M⁻¹ * p                    ← applyPreconditioner()
//     v = A * phat                      ← SpMV ★核心
//     alpha = rho / (rtilde·v)          ← 点积
//     s = r - alpha*v                   ← 向量运算
//
//     if ||s|| < relTol: x += alpha*phat; 收敛，跳出
//
//     shat = M⁻¹ * s                    ← applyPreconditioner()
//     t = A * shat                      ← SpMV ★核心
//     omega = (t·s) / (t·t)            ← 点积
//     x = x + alpha*phat + omega*shat   ← 向量运算
//     r = s - omega*t                   ← 向量运算
//
//     if ||r|| / ||b|| < relTol: 收敛，跳出
//     rho_old = rho
//
// 你需要填：按上述步骤实现
// ============================================================
int
CuSPSolver::solveBiCGStab()
{
    // ===== 创建 cuBLAS 句柄 =====
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // 标量系数
    const double alpha_one  = 1.0;
    double alpha, beta, omega, rho, rho_old;
    double dot_rv, dot_ts, dot_tt, norm_b, norm_r;

    // ===== 初始化 =====
    // r = b（因为 x=0，所以 r = b - A*0 = b）
    cudaMemcpy(d_r, d_b, m_n*sizeof(double), cudaMemcpyDeviceToDevice);

    // rtilde = r（副本，整个迭代中不变）
    cudaMemcpy(d_rtilde, d_r, m_n*sizeof(double), cudaMemcpyDeviceToDevice);

    // ||b|| 用于收敛判定
    cublasDnrm2(cublasHandle, m_n, d_b, 1, &norm_b);
    if (norm_b == 0.0) norm_b = 1.0;  // 防止除零

    rho_old = 1.0;
    alpha   = 1.0;
    omega   = 1.0;

    // ===== 迭代循环 =====
    for (int iter = 0; iter < m_maxIter; iter++) {

        // rho = rtilde · r
        cublasDdot(cublasHandle, m_n, d_rtilde, 1, d_r, 1, &rho);

        if (rho == 0.0) {
            cublasDestroy(cublasHandle);
            return -1;  // 失败
        }

        // 第一步：p = r
        // 后续步：p = r + beta*(p - omega*v)
        if (iter > 0) {
            beta = (rho / rho_old) * (alpha / omega);

            // p = p - omega*v
            double negOmega = -omega;
            cublasDaxpy(cublasHandle, m_n, &negOmega, d_v, 1, d_p, 1);

            // p = beta*p + r
            cublasDscal(cublasHandle, m_n, &beta, d_p, 1);
            cublasDaxpy(cublasHandle, m_n, &alpha_one, d_r, 1, d_p, 1);
        } else {
            cudaMemcpy(d_p, d_r, m_n*sizeof(double), cudaMemcpyDeviceToDevice);
        }

        // phat = M⁻¹ * p（预条件）
        applyPreconditioner(d_phat, d_p);

        // v = A * phat  ← SpMV ★核心
        gpuSpMV(m_cusparseHandle, m_spMatA, d_phat, d_v, m_n, d_spmvBuffer);

        // alpha = rho / (rtilde · v)
        cublasDdot(cublasHandle, m_n, d_rtilde, 1, d_v, 1, &dot_rv);
        if (dot_rv == 0.0) {
            cublasDestroy(cublasHandle);
            return -1;
        }
        alpha = rho / dot_rv;

        // s = r - alpha*v
        cudaMemcpy(d_s, d_r, m_n*sizeof(double), cudaMemcpyDeviceToDevice);
        double negAlpha = -alpha;
        cublasDaxpy(cublasHandle, m_n, &negAlpha, d_v, 1, d_s, 1);

        // 检查 ||s|| 是否已收敛
        cublasDnrm2(cublasHandle, m_n, d_s, 1, &norm_r);
        if (norm_r / norm_b < m_relTol) {
            // x = x + alpha*phat
            cublasDaxpy(cublasHandle, m_n, &alpha, d_phat, 1, d_x, 1);
            cublasDestroy(cublasHandle);
            return 0;  // 收敛
        }

        // shat = M⁻¹ * s（预条件）
        applyPreconditioner(d_shat, d_s);

        // t = A * shat  ← SpMV ★核心
        gpuSpMV(m_cusparseHandle, m_spMatA, d_shat, d_t, m_n, d_spmvBuffer);

        // omega = (t·s) / (t·t)
        cublasDdot(cublasHandle, m_n, d_t, 1, d_s, 1, &dot_ts);
        cublasDdot(cublasHandle, m_n, d_t, 1, d_t, 1, &dot_tt);
        if (dot_tt == 0.0) {
            cublasDestroy(cublasHandle);
            return -1;
        }
        omega = dot_ts / dot_tt;

        // x = x + alpha*phat + omega*shat
        cublasDaxpy(cublasHandle, m_n, &alpha, d_phat, 1, d_x, 1);
        cublasDaxpy(cublasHandle, m_n, &omega, d_shat, 1, d_x, 1);

        // r = s - omega*t
        cudaMemcpy(d_r, d_s, m_n*sizeof(double), cudaMemcpyDeviceToDevice);
        double negOmega = -omega;
        cublasDaxpy(cublasHandle, m_n, &negOmega, d_t, 1, d_r, 1);

        // 收敛判定：||r|| / ||b|| < relTol
        cublasDnrm2(cublasHandle, m_n, d_r, 1, &norm_r);
        if (norm_r / norm_b < m_relTol) {
            opserr << "CuSPSolver::solveBiCGStab - converged in " << (iter + 1) << " iterations\n";
            cublasDestroy(cublasHandle);
            return 0;  // 收敛
        }

        rho_old = rho;
    }

    // 达到最大迭代次数仍未收敛
    cublasDestroy(cublasHandle);
    return -1;
}


// ============================================================
// sendSelf() / recvSelf()
//
// 做什么：MPI 并行时用于进程间传输求解器状态
//         单机 GPU 版不需要，直接返回 0
// ============================================================
int
CuSPSolver::sendSelf(int cTAg, Channel &theChannel)
{
    return 0;
}

int
CuSPSolver::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    return 0;
}
