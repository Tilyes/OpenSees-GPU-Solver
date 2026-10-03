# s3_brick_transient.tcl — 场景3：3D 实体块，瞬态动力分析（Newmark 多步）
# 用法: OpenSees s3_brick_transient.tcl <solver> <nx> <ny> <nz> <nsteps> [dt]
#   solver: superlu | mumps | cusp-cg | cusp-bicgstab（rtol/maxIter 用默认值）
# 流程: 顶部集中力静力 1 步 → 去荷载 → Rayleigh 质量比例阻尼 → Newmark 自由振动 nsteps 步
# 输出: RESULT scenario=s3 solver=... np=... ndof=... t_gravity=... t_transient=... t_perstep=... tipDisp=...

set solver   [lindex $argv 0]
set nx       [lindex $argv 1]
set ny       [lindex $argv 2]
set nz       [lindex $argv 3]
set nsteps   [lindex $argv 4]
set dt       [expr {[llength $argv] > 5 ? [lindex $argv 5] : 0.02}]
set rtol     1e-8
set maxIter  20000

if {[catch {getNP} np]} { set np 1 }

set tStart [clock milliseconds]

wipe
model BasicBuilder -ndm 3 -ndf 3

# 弹性材料，带密度（瞬态需要质量矩阵）
nDMaterial ElasticIsotropic 1 100.0 0.25 1.27

set eleArgs "1"
set element stdBrick

set nn [expr {($nz+1)*($nx+1)*($ny+1)}]
set ndof [expr {3*$nn}]

block3D $nx $ny $nz 1 1 $element $eleArgs {
    1   -1     -1      0
    2    1     -1      0
    3    1      1      0
    4   -1      1      0
    5   -1     -1     10
    6    1     -1     10
    7    1      1     10
    8   -1      1     10
}

fixZ 0.0 1 1 1
pattern Plain 1 Linear {
    load $nn 0.10 0.10 0.0
}

proc selectSolver {solver rtol maxIter} {
    if {$solver eq "cusp-cg"} {
        system cusp -pre jacobi -solver cg -rtol $rtol -maxIter $maxIter
    } elseif {$solver eq "cusp-bicgstab"} {
        system cusp -pre jacobi -solver bicgstab -rtol $rtol -maxIter $maxIter
    } elseif {$solver eq "cusp-ilu0-cg"} {
        system cusp -pre ilu0 -solver cg -rtol $rtol -maxIter $maxIter
    } elseif {$solver eq "cusp-ilu0-bicgstab"} {
        system cusp -pre ilu0 -solver bicgstab -rtol $rtol -maxIter $maxIter
    } elseif {$solver eq "mumps"} {
        system Mumps
    } else {
        system SuperLU
    }
}

# —— 静力重力步 ——
selectSolver $solver $rtol $maxIter
constraints Plain
numberer RCM
test NormUnbalance 1.0e-8 20 0
algorithm Newton
integrator LoadControl 1.0 1
analysis Static

set ok [analyze 1]
if {$ok != 0} {
    puts "RESULT scenario=s3 solver=$solver np=$np ndof=$ndof GRAVITY_FAILED"
    exit 1
}

set tGravity [clock milliseconds]

# —— 瞬态自由振动 ——
wipeAnalysis
setTime 0.0
remove loadPattern 1

rayleigh 0.01 0.0 0.0 0.0

test EnergyIncr 1.0e-8 20 0
algorithm Newton
numberer RCM
selectSolver $solver $rtol $maxIter
constraints Plain
integrator Newmark 0.5 0.25
analysis Transient

set tTransStart [clock milliseconds]

set ok [analyze $nsteps $dt]

set tTransient [clock milliseconds]

if {$ok != 0} {
    puts "RESULT scenario=s3 solver=$solver np=$np ndof=$ndof TRANSIENT_FAILED at step"
    exit 1
}

set tipDisp [lindex [nodeDisp $nn] 0]
set tTran [expr {($tTransient-$tTransStart)/1000.0}]
puts "RESULT scenario=s3 solver=$solver np=$np ndof=$ndof nsteps=$nsteps t_gravity=[expr {($tGravity-$tStart)/1000.0}] t_transient=$tTran t_perstep=[expr {$tTran/$nsteps}] tipDisp=$tipDisp"
exit 0
