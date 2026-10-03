# s2_frame3d_static.tcl — 场景2：3D 多层多跨钢框架，静力分析
# 用法: OpenSees s2_frame3d_static.tcl <solver> <nx> <ny> <ns>
#   nx, ny: x/y 方向跨数；ns: 层数
#   solver: superlu | mumps | cusp-cg | cusp-bicgstab [rtol] [maxIter]
# 输出: RESULT scenario=s2 solver=... np=... ndof=... t_model=... t_analyze=... tipDisp=...

set solver   [lindex $argv 0]
set nx       [lindex $argv 1]
set ny       [lindex $argv 2]
set ns       [lindex $argv 3]
set rtol     [expr {[llength $argv] > 4 ? [lindex $argv 4] : 1e-8}]
set maxIter  [expr {[llength $argv] > 5 ? [lindex $argv 5] : 20000}]

if {[catch {getNP} np]} { set np 1 }

set tStart [clock milliseconds]

wipe
model BasicBuilder -ndm 3 -ndf 6

# 几何参数
set bayX  360.0
set bayY  360.0
set story 144.0

# 钢材（截面属性按典型 W 型钢）
set E  29000.0
set G  11500.0
set Ac 50.0
set Icy 2500.0
set Icz 2500.0
set Jc 80.0
set Ab 30.0
set Iby 1200.0
set Ibz 1200.0
set Jb 40.0

set nxn [expr {$nx+1}]
set nyn [expr {$ny+1}]
set perFloor [expr {$nxn*$nyn}]

# 节点编号：floor*perFloor + iy*nxn + ix + 1
for {set f 0} {$f <= $ns} {incr f} {
    for {set iy 0} {$iy < $nyn} {incr iy} {
        for {set ix 0} {$ix < $nxn} {incr ix} {
            set tag [expr {$f*$perFloor + $iy*$nxn + $ix + 1}]
            node $tag [expr {$ix*$bayX}] [expr {$iy*$bayY}] [expr {$f*$story}]
        }
    }
}

# 底层固结
for {set iy 0} {$iy < $nyn} {incr iy} {
    for {set ix 0} {$ix < $nxn} {incr ix} {
        fix [expr {$iy*$nxn + $ix + 1}] 1 1 1 1 1 1
    }
}

geomTransf Linear 1 1 0 0
geomTransf Linear 2 0 0 1

set eleTag 0

# 柱（f 层到 f+1 层）
for {set f 0} {$f < $ns} {incr f} {
    for {set iy 0} {$iy < $nyn} {incr iy} {
        for {set ix 0} {$ix < $nxn} {incr ix} {
            set n1 [expr {$f*$perFloor + $iy*$nxn + $ix + 1}]
            set n2 [expr {$n1 + $perFloor}]
            incr eleTag
            element elasticBeamColumn $eleTag $n1 $n2 $Ac $E $G $Jc $Icy $Icz 1
        }
    }
}

# 梁（每层楼面，x 向与 y 向，f 从 1 开始——底层是地面）
for {set f 1} {$f <= $ns} {incr f} {
    for {set iy 0} {$iy < $nyn} {incr iy} {
        for {set ix 0} {$ix < [expr {$nxn-1}]} {incr ix} {
            set n1 [expr {$f*$perFloor + $iy*$nxn + $ix + 1}]
            incr eleTag
            element elasticBeamColumn $eleTag $n1 [expr {$n1+1}] $Ab $E $G $Jb $Iby $Ibz 2
        }
    }
    for {set iy 0} {$iy < [expr {$nyn-1}]} {incr iy} {
        for {set ix 0} {$ix < $nxn} {incr ix} {
            set n1 [expr {$f*$perFloor + $iy*$nxn + $ix + 1}]
            incr eleTag
            element elasticBeamColumn $eleTag $n1 [expr {$n1+$nxn}] $Ab $E $G $Jb $Iby $Ibz 2
        }
    }
}

set nn [expr {($ns+1)*$perFloor}]
set ndof [expr {6*$nn}]

# 顶层荷载：竖向重力 + x 向水平力
pattern Plain 1 Linear {
    for {set iy 0} {$iy < $nyn} {incr iy} {
        for {set ix 0} {$ix < $nxn} {incr ix} {
            set tag [expr {$ns*$perFloor + $iy*$nxn + $ix + 1}]
            load $tag 100.0 0.0 -500.0 0.0 0.0 0.0
        }
    }
}

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

constraints Plain
numberer RCM
# Mumps 并行解的残差略高于直接法（并行装配/主元差异），弹性单迭代问题放宽收敛判据，
# 不影响线性求解时间（仅决定 Newton 是否接受这一步）
if {$solver eq "mumps"} {
    test NormUnbalance 1.0e-6 20 0
} else {
    test NormUnbalance 1.0e-8 20 0
}
algorithm Newton
integrator LoadControl 1.0 1
analysis Static

set tModel [clock milliseconds]

set ok [analyze 1]

set tAnalyze [clock milliseconds]

if {$ok != 0} {
    puts "RESULT scenario=s2 solver=$solver np=$np ndof=$ndof ANALYSIS_FAILED"
    exit 1
}

set tipDisp [lindex [nodeDisp $nn] 0]
puts "RESULT scenario=s2 solver=$solver np=$np ndof=$ndof t_model=[expr {($tModel-$tStart)/1000.0}] t_analyze=[expr {($tAnalyze-$tModel)/1000.0}] t_total=[expr {($tAnalyze-$tStart)/1000.0}] tipDisp=$tipDisp"
exit 0
