#!/bin/bash
# run_remaining.sh — 补跑被暂停/超时的 Mumps 配置，总时长上限约 55 分钟
cd "$(dirname "$0")"
SP=../build/OpenSeesSP

run() {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] RUN  $name"
    timeout "$1" mpiexec -np "$2" $SP "$3" mumps $4 > "results/$name.log" 2>&1
    local rc=$?
    if ! grep -q "^RESULT" "results/$name.log"; then
        echo "RESULT DNF rc=$rc" >> "results/$name.log"
        echo "[$(date +%H:%M:%S)] DNF  $name (rc=$rc)"
    else
        grep "^RESULT" "results/$name.log" | tail -1
    fi
}

# S3 瞬态（10x10x40, 200 步）—— 从未运行，优先
run s3_sp_mumps_np4 1200 4 models/s3_brick_transient.tcl "10 10 40 200"
run s3_sp_mumps_np8 1200 8 models/s3_brick_transient.tcl "10 10 40 200"

# S2 框架静力 —— 已两次 30 分钟超时，15 分钟复验确认
run s2_sp_mumps_np4 900 4 models/s2_frame3d_static.tcl "16 16 24"

echo "[$(date +%H:%M:%S)] ALL DONE"
