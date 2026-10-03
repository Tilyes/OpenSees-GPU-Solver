#!/bin/bash
# calibrate.sh — 规模校准记录（2026-08-10 已执行，供复现参考）
# 目标：串行 SuperLU 单次求解 2–6 分钟，且不超出 15GB WSL 内存
cd "$(dirname "$0")"
SERIAL=../build/OpenSees
CUDA=../build-cuda/OpenSees
SP=../build/OpenSeesSP
LIBCUDA=/usr/lib/wsl/lib/libcuda.so.1

# —— S1 规模阶梯（实测）——
# 10x10x100  ->  36,663 DOF: SuperLU   3.5s
# 15x15x160  -> 123,648 DOF: SuperLU  62.6s / cusp-CG 2.5s / Mumps np8 18.4s
# 18x18x200  -> 217,683 DOF: SuperLU 201.7s / cusp-CG 4.7s   ★ 采用
$SERIAL models/s1_brick_static.tcl superlu 10 10 100
env LD_PRELOAD=$LIBCUDA $CUDA models/s1_brick_static.tcl cusp-cg 15 15 160
mpiexec -np 8 $SP models/s1_brick_static.tcl mumps 15 15 160

# —— S2 规模阶梯（实测）——
# 20x20x30   ->  82,026 DOF: SuperLU 479.5s（超预算）
# 16x16x24   ->  43,350 DOF: SuperLU 107.5s / cusp-CG 0.57s  ★ 采用
$SERIAL models/s2_frame3d_static.tcl superlu 16 16 24

# —— S3 规模阶梯（实测，per-step）——
# 8x8x40     ->   9,963 DOF: 0.887s/step
# 12x12x80   ->  41,067 DOF: 10.93s/step（太慢）
# 10x10x40   ->  14,883 DOF: SuperLU 2.29s/step / cusp-CG 0.20s/step ★ 采用（200 步）
$SERIAL models/s3_brick_transient.tcl superlu 10 10 40 10
env LD_PRELOAD=$LIBCUDA $CUDA models/s3_brick_transient.tcl cusp-cg 10 10 40 10
