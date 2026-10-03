#!/bin/bash
# run_hypersp.sh — 用 SCOTCH 版 Mumps（build-hypersp）补测各场景
cd "$(dirname "$0")"
SP=../build-hypersp/OpenSeesSP

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

run s2_sp_mumps_np8_scotch 1200 8 models/s2_frame3d_static.tcl "16 16 24"
run s1_sp_mumps_np4_scotch 1200 4 models/s1_brick_static.tcl "18 18 200"
run s1_sp_mumps_np8_scotch 1200 8 models/s1_brick_static.tcl "18 18 200"
run s3_sp_mumps_np4_scotch 1200 4 models/s3_brick_transient.tcl "10 10 40 200"
run s3_sp_mumps_np8_scotch 1200 8 models/s3_brick_transient.tcl "10 10 40 200"

echo "[$(date +%H:%M:%S)] ALL DONE"
