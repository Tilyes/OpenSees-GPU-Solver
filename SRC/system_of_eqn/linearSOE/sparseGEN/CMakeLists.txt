#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#==============================================================================

target_sources(OPS_SysOfEqn
  PRIVATE 
    SparseGenColLinSOE.cpp
    SparseGenColLinSolver.cpp
    SparseGenRowLinSOE.cpp
    SparseGenRowLinSolver.cpp
    SuperLU.cpp
  PUBLIC
    SparseGenColLinSOE.h
    SparseGenColLinSolver.h
    SparseGenRowLinSOE.h
    SparseGenRowLinSolver.h
    SuperLU.h
)

target_sources(OPS_PFEM
    PRIVATE
        PFEMCompressibleLinSOE.cpp
        PFEMCompressibleSolver.cpp
        PFEMDiaLinSOE.cpp
        PFEMDiaSolver.cpp
        PFEMLinSOE.cpp
        PFEMQuasiLinSOE.cpp
        PFEMQuasiSolver.cpp
        PFEMSolver.cpp
        PFEMSolver_Laplace.cpp
        PFEMSolver_LumpM.cpp
        PFEMSolver_LumpM.cpp
        PFEMSolver_Umfpack.cpp
        PFEMSolver_Mumps.cpp	
    PUBLIC
        PFEMCompressibleLinSOE.h
        PFEMCompressibleSolver.h
        PFEMDiaLinSOE.h
        PFEMDiaSolver.h
        PFEMLinSOE.h
        PFEMQuasiLinSOE.h
        PFEMQuasiSolver.h
        PFEMSolver.h
        PFEMSolver_Laplace.h
        PFEMSolver_LumpM.h
        PFEMSolver_LumpM.h
        PFEMSolver_Umfpack.h
        PFEMSolver_Mumps.h	
)

if(MPI_FOUND)

target_sources(OpenSeesMP
    PRIVATE 
        DistributedSuperLU.cpp
        DistributedSparseGenColLinSOE.cpp
        DistributedSparseGenRowLinSOE.cpp
        PFEMSolver_Mumps.cpp
        PFEMSolver_LumpM.cpp
        PFEMCompressibleSolver_Mumps.cpp
    PUBLIC
        DistributedSuperLU.h
        DistributedSparseGenColLinSOE.h
        DistributedSparseGenRowLinSOE.h
        PFEMSolver_Mumps.h
        PFEMCompressibleSolver_Mumps.h
)

target_sources(OpenSeesSP
    PRIVATE 
        DistributedSuperLU.cpp
        DistributedSparseGenColLinSOE.cpp
        DistributedSparseGenRowLinSOE.cpp
        PFEMSolver_Mumps.cpp
        PFEMSolver_LumpM.cpp
        PFEMCompressibleSolver_Mumps.cpp
    PUBLIC
        DistributedSuperLU.h
        DistributedSparseGenColLinSOE.h
        DistributedSparseGenRowLinSOE.h
        PFEMSolver_Mumps.h
        PFEMCompressibleSolver_Mumps.h
)

endif()


target_include_directories(OPS_SysOfEqn PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# ===== CUDA 求解器（可选）=====
# 检测 CUDA Toolkit，有则编译 CuSPSolver，无则跳过
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION} — enabling CuSPSolver")
    # 用 CUDA 编译器编译 CuSPSolver.cpp（因为含 <<<>>> kernel 语法）
    # TARGET_DIRECTORY 必须加：OPS_SysOfEqn 在上级目录创建，
    # 不加的话 LANGUAGE 属性只在本目录作用域生效，会被上级目标忽略
    set_source_files_properties(CuSPSolver.cpp
        TARGET_DIRECTORY OPS_SysOfEqn
        PROPERTIES LANGUAGE CUDA)
    target_sources(OPS_SysOfEqn PRIVATE CuSPSolver.cpp)
    target_compile_definitions(OPS_SysOfEqn PRIVATE _CUSP)
    target_link_libraries(OPS_SysOfEqn PRIVATE
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
    )
endif()

