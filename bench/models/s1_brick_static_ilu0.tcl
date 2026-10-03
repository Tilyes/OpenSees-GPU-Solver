# s1_brick_static.tcl — 场景1：3D 实体悬臂块，静力分析
# 用法: OpenSees s1_brick_static.tcl <solver> <nx> <ny> <nz>
#   solver: superlu | mumps | cusp-cg | cusp-bicgstab [rtol] [maxIter]
# 输出: RESULT scenario=s1 solver=... np=... ndof=... t_model=... t_analyze=... tipDisp=...

set solver   [lindex $argv 0]
set nx       [lindex $argv 1]
set ny       [lindex $argv 2]
set nz       [lindex $argv 3]
set rtol     [expr {[llength $argv] > 4 ? [lindex $argv 4] : 1e-8}]
set maxIter  [expr {[llength $argv] > 5 ? [lindex $argv 5] : 20000}]

if {[catch {getNP} np]} { set np 1 }

set tStart [clock milliseconds]

wipe
model BasicBuilder -ndm 3 -ndf 3

# 弹性材料（SPD 刚度矩阵，CG 友好）
nDMaterial ElasticIsotropic 1 100.0 0.25 1.27

set eleArgs "1"
set element stdBrick

set nn [expr {($nz+1)*($nx+1)*($ny+1)}]
set ndof [expr {3*$nn}]

# 网格：悬臂块，z 向为长度方向
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

# 固定底面，顶部节点集中力
fixZ 0.0 1 1 1
pattern Plain 1 Linear {
    load $nn 0.10 0.10 0.0
}

# 求解器选择
if {$solver eq "cusp-cg"} {
    system cusp -pre jacobi -solver cg -rtol $rtol -maxIter $maxIter
} elseif {$solver eq "cusp-ilu0-cg"} {
    system cusp -pre ilu0 -solver cg -rtol $rtol -maxIter $maxIter
} elseif {$solver eq "cusp-ilu0-bicgstab"} {
    system cusp -pre ilu0 -solver bicgstab -rtol $rtol -maxIter $maxIter
} elseif {$solver eq "cusp-bicgstab"} {
    system cusp -pre jacobi -solver bicgstab -rtol $rtol -maxIter $maxIter
} elseif {$solver eq "mumps"} {
    system Mumps
} else {
    system SuperLU
}

constraints Plain
numberer RCM
test NormUnbalance 1.0e-8 20 0
algorithm Newton
integrator LoadControl 1.0 1
analysis Static

set tModel [clock milliseconds]

set ok [analyze 1]

set tAnalyze [clock milliseconds]

if {$ok != 0} {
    puts "RESULT scenario=s1 solver=$solver np=$np ndof=$ndof ANALYSIS_FAILED"
    exit 1
}

set tipDisp [lindex [nodeDisp $nn] 0]
puts "RESULT scenario=s1 solver=$solver np=$np ndof=$ndof t_model=[expr {($tModel-$tStart)/1000.0}] t_analyze=[expr {($tAnalyze-$tModel)/1000.0}] t_total=[expr {($tAnalyze-$tStart)/1000.0}] tipDisp=$tipDisp"
exit 0
