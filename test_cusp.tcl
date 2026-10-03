# test_cusp.tcl — 阶段 A5/A6：对比 GPU cusp 与 CPU SuperLU 的求解结果
# 用法: OpenSees test_cusp.tcl [cusp|superlu]
#
# 模型：2D 悬臂梁，10 个弹性梁单元，端部竖向集中力
# 解析解（欧拉-伯努利）：端部挠度 dy = P*L^3/(3*E*I) = 1000*100^3/(3*29000*1000)
#                     转角 dm = P*L^2/(2*E*I)

set solverType [lindex $argv 0]
if {$solverType eq ""} { set solverType "cusp" }

wipe
model BasicBuilder -ndm 2 -ndf 3

set L 100.0
set nEle 10
set E 29000.0
set A 10.0
set I 1000.0
set P -1000.0

for {set i 0} {$i <= $nEle} {incr i} {
    node [expr {$i+1}] [expr {$i*$L/$nEle}] 0.0
}
fix 1 1 1 1

geomTransf Linear 1
for {set i 1} {$i <= $nEle} {incr i} {
    element elasticBeamColumn $i $i [expr {$i+1}] $A $E $I 1
}

pattern Plain 1 Linear {
    load [expr {$nEle+1}] 0.0 $P 0.0
}

if {$solverType eq "cusp"} {
    system cusp -pre jacobi -solver cg -rtol 1e-10 -maxIter 10000
} else {
    system SuperLU
}
constraints Plain
numberer Plain
algorithm Linear
integrator LoadControl 1.0
analysis Static

set ok [analyze 1]
if {$ok != 0} {
    puts "RESULT $solverType ANALYSIS_FAILED"
    exit 1
}

set tip [expr {$nEle+1}]
set disp [nodeDisp $tip]
set dyExact [expr {$P*$L*$L*$L/(3.0*$E*$I)}]
puts "RESULT $solverType tipDisp = $disp"
puts "RESULT $solverType exact_dy = $dyExact"
exit 0
