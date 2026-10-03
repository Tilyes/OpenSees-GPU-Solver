// spsv_repro.cu — 复现 cusparseSpSV_solve FPE
// 流程与 CuSPSolver::factorizeILU0 + applyPreconditioner 完全一致：
//   csrilu02 分解 → 共享 LU 值的 L/U generic 描述符 → SpSV analysis →
//   solve 时用【全新】DnVec 描述符（模式 A）或【复用】描述符（模式 B）
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do { cusparseStatus_t s_ = (x); \
    if (s_ != CUSPARSE_STATUS_SUCCESS) { \
        printf("%s:%d cusparse error %d\n", __FILE__, __LINE__, (int)s_); \
    } } while(0)

int main(int argc, char** argv)
{
    int nx = argc > 1 ? atoi(argv[1]) : 60;   // 每维网格数
    int mode = argc > 2 ? atoi(argv[2]) : 0;  // 0=fresh描述符(现实现) 1=复用描述符
    int n = nx * nx * nx;
    printf("n=%d mode=%d\n", n, mode);

    // 3D 7 点 Laplacian（SPD），CSR
    std::vector<int> rowPtr(n + 1, 0);
    for (int k = 0; k < nx; k++)
      for (int j = 0; j < nx; j++)
        for (int i = 0; i < nx; i++) {
            int r = (k*nx + j)*nx + i;
            int cnt = 1;
            if (i > 0) cnt++; if (i < nx-1) cnt++;
            if (j > 0) cnt++; if (j < nx-1) cnt++;
            if (k > 0) cnt++; if (k < nx-1) cnt++;
            rowPtr[r+1] = rowPtr[r] + cnt;
        }
    int nnz = rowPtr[n];
    std::vector<int> colInd(nnz);
    std::vector<double> val(nnz);
    for (int k = 0, p = 0; k < nx; k++)
      for (int j = 0; j < nx; j++)
        for (int i = 0; i < nx; i++) {
            int r = (k*nx + j)*nx + i;
            if (i > 0)    { colInd[p] = r-1;      val[p++] = -1; }
            if (j > 0)    { colInd[p] = r-nx;     val[p++] = -1; }
            if (k > 0)    { colInd[p] = r-nx*nx;  val[p++] = -1; }
            colInd[p] = r; val[p++] = 6.1;
            if (k < nx-1) { colInd[p] = r+nx*nx;  val[p++] = -1; }
            if (j < nx-1) { colInd[p] = r+nx;     val[p++] = -1; }
            if (i < nx-1) { colInd[p] = r+1;      val[p++] = -1; }
        }
    printf("nnz=%d\n", nnz);

    double *d_valLU, *d_b, *d_w, *d_x;
    int *d_rowPtr, *d_colInd;
    cudaMalloc(&d_valLU, nnz*sizeof(double));
    cudaMalloc(&d_rowPtr, (n+1)*sizeof(int));
    cudaMalloc(&d_colInd, nnz*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(double));
    cudaMalloc(&d_w, n*sizeof(double));
    cudaMalloc(&d_x, n*sizeof(double));
    cudaMemcpy(d_valLU, val.data(), nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, rowPtr.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, colInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);

    cusparseHandle_t h; CK(cusparseCreate(&h));
    cusparseMatDescr_t md; CK(cusparseCreateMatDescr(&md));
    cusparseSetMatType(md, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(md, CUSPARSE_INDEX_BASE_ZERO);

    // ILU0（mode 2 跳过，直接用原矩阵测 SpSV）
    if (mode != 2) {
    csrilu02Info_t info; CK(cusparseCreateCsrilu02Info(&info));
    int bufSize = 0;
    CK(cusparseDcsrilu02_bufferSize(h, n, nnz, md, d_valLU, d_rowPtr, d_colInd, info, &bufSize));
    void* d_buf; cudaMalloc(&d_buf, bufSize);
    CK(cusparseDcsrilu02_analysis(h, n, nnz, md, d_valLU, d_rowPtr, d_colInd, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));
    CK(cusparseDcsrilu02(h, n, nnz, md, d_valLU, d_rowPtr, d_colInd, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));
    printf("ilu0 done\n");
    // 完整性检查：结构是否被改写、值是否有限
    {
        std::vector<int> rp2(n+1), ci2(nnz);
        std::vector<double> v2(nnz);
        cudaMemcpy(rp2.data(), d_rowPtr, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(ci2.data(), d_colInd, nnz*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(v2.data(), d_valLU, nnz*sizeof(double), cudaMemcpyDeviceToHost);
        int bad = 0, badv = 0;
        for (int i = 0; i <= n; i++) if (rp2[i] != rowPtr[i]) bad++;
        for (int i = 0; i < nnz; i++) if (ci2[i] != colInd[i]) bad++;
        for (int i = 0; i < nnz; i++) if (!isfinite(v2[i])) badv++;
        double mn = 1e300, mx = 0;
        for (int r = 0; r < n; r++)  // 每行对角线（每行第4个非零元是 r 自身）
            for (int p = rowPtr[r]; p < rowPtr[r+1]; p++)
                if (colInd[p] == r) { double a = fabs(v2[p]); if (a<mn) mn=a; if (a>mx) mx=a; }
        printf("check: structBad=%d nonFinite=%d diagAbs=[%g, %g]\n", bad, badv, mn, mx);
    }
    } else {
    printf("mode 2: skip ilu0, raw matrix SpSV\n");
    }

    // L/U 描述符
    cusparseSpMatDescr_t L, U;
    CK(cusparseCreateCsr(&L, n, n, nnz, d_rowPtr, d_colInd, d_valLU,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    CK(cusparseSpMatSetAttribute(L, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)));
    CK(cusparseSpMatSetAttribute(L, CUSPARSE_SPMAT_DIAG_TYPE, &unit, sizeof(unit)));
    CK(cusparseCreateCsr(&U, n, n, nnz, d_rowPtr, d_colInd, d_valLU,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CK(cusparseSpMatSetAttribute(U, CUSPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper)));
    CK(cusparseSpMatSetAttribute(U, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit, sizeof(nonunit)));

    cusparseSpSVDescr_t svL, svU;
    CK(cusparseSpSV_createDescr(&svL));
    CK(cusparseSpSV_createDescr(&svU));

    const double one = 1.0;
    cusparseDnVecDescr_t vecB, vecW, vecX;
    CK(cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_64F));
    CK(cusparseCreateDnVec(&vecW, n, d_w, CUDA_R_64F));
    CK(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));

    size_t sbL = 0, sbU = 0;
    CK(cusparseSpSV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, L, vecB, vecW,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, svL, &sbL));
    void* d_sbL; cudaMalloc(&d_sbL, sbL);
    CK(cusparseSpSV_analysis(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, L, vecB, vecW,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, svL, d_sbL));
    CK(cusparseSpSV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, U, vecW, vecX,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, svU, &sbU));
    void* d_sbU; cudaMalloc(&d_sbU, sbU);
    CK(cusparseSpSV_analysis(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, U, vecW, vecX,
        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, svU, d_sbU));
    printf("analysis done (bufL=%zu bufU=%zu)\n", sbL, sbU);

    if (mode == 0) { CK(cusparseDestroyDnVec(vecB)); CK(cusparseDestroyDnVec(vecW)); CK(cusparseDestroyDnVec(vecX)); }

    // solve：与 applyPreconditioner 相同，每次迭代调一次
    for (int it = 0; it < 3; it++) {
        cusparseDnVecDescr_t vI, vW, vO;
        if (mode == 0) {
            CK(cusparseCreateDnVec(&vI, n, d_b, CUDA_R_64F));
            CK(cusparseCreateDnVec(&vW, n, d_w, CUDA_R_64F));
            CK(cusparseCreateDnVec(&vO, n, d_x, CUDA_R_64F));
        } else { vI = vecB; vW = vecW; vO = vecX; }
        printf("iter %d: solve L...\n", it); fflush(stdout);
        CK(cusparseSpSV_solve(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, L, vI, vW,
            CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, svL));
        printf("iter %d: solve U...\n", it); fflush(stdout);
        CK(cusparseSpSV_solve(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, U, vW, vO,
            CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, svU));
        if (mode == 0) { CK(cusparseDestroyDnVec(vI)); CK(cusparseDestroyDnVec(vW)); CK(cusparseDestroyDnVec(vO)); }
        cudaDeviceSynchronize();
        printf("iter %d done\n", it);
    }
    printf("ALL OK\n");
    return 0;
}
