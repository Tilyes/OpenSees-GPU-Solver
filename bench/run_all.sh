#!/bin/bash
# run_all.sh — 全量基准测试：3 场景 × {serial-SuperLU, cuda-cusp, SP-Mumps np4/np8}
# 用法: bash run_all.sh   （在 bench/ 目录下运行；预计 25–40 分钟）

cd "$(dirname "$0")"
mkdir -p results

SERIAL=../build/OpenSees
CUDA=../build-cuda/OpenSees
SP=../build/OpenSeesSP
LIBCUDA=/usr/lib/wsl/lib/libcuda.so.1

# 场景规模（校准结果，2026-08-10）
S1="18 18 200"     # 217,683 DOF，静力
S2="16 16 24"      #  43,350 DOF，静力
S3="10 10 40 200"  #  14,883 DOF，瞬态 200 步

run() {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] RUN  $name"
    timeout 1800 "$@" > "results/$name.log" 2>&1
    local rc=$?
    if ! grep -q "^RESULT" "results/$name.log"; then
        echo "RESULT DNF rc=$rc" >> "results/$name.log"
        echo "[$(date +%H:%M:%S)] DNF  $name (rc=$rc)"
    else
        grep "^RESULT" "results/$name.log" | tail -1
    fi
}

# ---------- 场景 1：3D 实体悬臂块·静力 ----------
run s1_serial_superlu $SERIAL models/s1_brick_static.tcl superlu $S1
run s1_cuda_cg        env LD_PRELOAD=$LIBCUDA $CUDA models/s1_brick_static.tcl cusp-cg $S1
run s1_sp_mumps_np4   mpiexec -np 4 $SP models/s1_brick_static.tcl mumps $S1
run s1_sp_mumps_np8   mpiexec -np 8 $SP models/s1_brick_static.tcl mumps $S1

# ---------- 场景 2：3D 多层钢框架·静力 ----------
run s2_serial_superlu $SERIAL models/s2_frame3d_static.tcl superlu $S2
run s2_cuda_cg        env LD_PRELOAD=$LIBCUDA $CUDA models/s2_frame3d_static.tcl cusp-cg $S2
run s2_sp_mumps_np4   mpiexec -np 4 $SP models/s2_frame3d_static.tcl mumps $S2
run s2_sp_mumps_np8   mpiexec -np 8 $SP models/s2_frame3d_static.tcl mumps $S2

# ---------- 场景 3：3D 实体块·瞬态 200 步 ----------
run s3_serial_superlu $SERIAL models/s3_brick_transient.tcl superlu $S3
run s3_cuda_cg        env LD_PRELOAD=$LIBCUDA $CUDA models/s3_brick_transient.tcl cusp-cg $S3
run s3_sp_mumps_np4   mpiexec -np 4 $SP models/s3_brick_transient.tcl mumps $S3
run s3_sp_mumps_np8   mpiexec -np 8 $SP models/s3_brick_transient.tcl mumps $S3

echo "[$(date +%H:%M:%S)] ALL DONE"
python3 summarize.py
