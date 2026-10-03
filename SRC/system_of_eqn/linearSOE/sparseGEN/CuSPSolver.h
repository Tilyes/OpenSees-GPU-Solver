#ifndef CuSPSolver_h
#define CuSPSolver_h

#include <SparseGenRowLinSolver.h>
#include <SparseGenRowLinSOE.h>

// CUDA 头文件
#include <cuda_runtime.h>
#include <cusparse.h>

class CuSPSolver : public SparseGenRowLinSolver
{
public:
    CuSPSolver(int maxIter = 5000, double relTol = 1e-6,
               int preCond = 1, int solverType = 1);
    ~CuSPSolver(void);

    int solve(void);
    int setSize(void);

    int sendSelf(int commitTag, Channel &theChannel);
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);
    int setLinearSOE(SparseGenRowLinSOE &theLinearSOE);

private:
    // 参数
    int    m_maxIter;      // 最大迭代次数
    double m_relTol;       // 相对收敛容差
    int    m_preCond;      // 0=none, 1=jacobi, 2=ilu0
    int    m_solverType;   // 0=cg, 1=bicgstab, 2=gmres

    // GPU 资源（在 setSize 中分配，析构中释放）
    cusparseHandle_t    m_cusparseHandle;
    cusparseMatDescr_t  m_matDescr;
    // 三角求解分析信息：csrsv2 已从 CUDA 12 移除，阶段 B 改用
    // generic API（cusparseSpSVDescr_t）

    // SpMV 用的 generic API 描述符（csrmv 已从 CUDA 12 移除）
    cusparseSpMatDescr_t m_spMatA;   // CSR 矩阵描述符（引用 d_csr* 指针）
    void   *d_spmvBuffer;            // SpMV 工作缓冲
    size_t  m_spmvBufferSize;

    // GPU 端 CSR 矩阵数据
    double *d_csrVal;     // 非零值
    int    *d_csrRowPtr;  // 行指针
    int    *d_csrColInd;  // 列索引
    double *d_b;          // 右端项
    double *d_x;          // 解向量

    // ILU0 分解后的 L/U 值（原地存储在 d_csrVal 中，
    // 但需要额外的 d_csrValLU 副本用于每次求解时恢复）
    double *d_csrValLU;   // ILU0 分解结果
    int    *d_rowPtrLU;   // ILU0 专用行指针副本（csrilu02 可能越界写，
    int    *d_colIndLU;   // 不能让它碰 SpMV 共用的 d_csrRowPtr/d_csrColInd）

    // 预条件子的对角线（Jacobi 用）
    double *d_diagInv;

    // 阶段 B：ILU0 + 自实现 level-scheduled 三角求解
    // cusparseDcsrilu02 与 cusparseSpSV 在 sm_120(Blackwell) 上均间歇性崩
    // （INVALID_VALUE / 越界写 / SIGFPE），故分解与三角求解都自实现。
    double *d_w;                   // 预条件中间向量（避免与调用方缓冲别名）
    int    *d_fwdOrder;            // 前推行序（按 level 排序）
    int    *d_bwdOrder;            // 后推行序（按 level 排序）
    int    *d_fwdLevelPtr;         // 前推各 level 行区间（device，长度 numFwdLevels+1）
    int    *d_bwdLevelPtr;         // 后推各 level 行区间（device，长度 numBwdLevels+1）
    int    *m_fwdLevelPtr;         // 前推各 level 行区间（host，长度 numFwdLevels+1）
    int    *m_bwdLevelPtr;         // 后推各 level 行区间（host，长度 numBwdLevels+1）
    int     m_numFwdLevels;        // 前推 level 数
    int     m_numBwdLevels;        // 后推 level 数
    bool    m_levelsBuilt;         // level 调度是否已构建
    int     m_coopGrid;            // cooperative 三角求解的 grid 大小（首次分解时算一次）
    int     m_coopBlock;           // cooperative 三角求解的 block 大小

    // 阶段 B：CPU 端 ILU0 工作区
    // cusparseDcsrilu02 在 sm_120(Blackwell) 上对大矩阵会越界写结构并清空对角线，
    // 改为 CPU 端做等价 ILU(0) 分解后把 L/U 值上传 GPU（三角求解仍走 SpSV）。
    double *h_valLU;   // CPU 分解缓冲 (nnz)
    double *h_w;       // CPU 稠密工作向量 (n)
    double *h_diagU;   // CPU U 对角线 (n)
    int    *h_marker;  // CPU 标记数组 (n)

    // BiCGStab 工作空间
    double *d_r, *d_rtilde, *d_p, *d_phat, *d_s, *d_shat, *d_t, *d_v;
    double *d_tmp;        // 通用临时向量

    // 状态
    int m_n;              // 矩阵阶数
    int m_nnz;            // 非零元素数
    bool m_initialized;   // GPU 资源是否已分配
    bool m_structureUploaded;  // CSR 结构（行指针/列索引/描述符）是否已上传
    bool m_factorized;    // 本步 ILU0 是否分解成功（false 时 ILU0 分支回退 Jacobi）

    // 私有方法
    int allocateGPU();
    int freeGPU();
    int factorizeILU0();
    int solveBiCGStab();
    int solveCG();
    int applyPreconditioner(double *d_x, double *d_b);  // M^{-1} * b
};

#endif