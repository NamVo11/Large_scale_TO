cmake_minimum_required(VERSION 3.18)
project(stage1_petsc_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED CXX)
find_package(OpenMP REQUIRED)
set(PETSC_PKG "PETSc" CACHE STRING "PETSc pkg-config package, e.g. petsc-dmo on MSYS2 UCRT64")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET ${PETSC_PKG})

add_executable(stage1_petsc_backend
  cpp/src/stage1/main.cpp
  cpp/src/stage1/IO.cpp
  cpp/src/stage1/FEMH8.cpp
  cpp/src/stage1/DynamicElasticSolver.cpp
)

target_include_directories(stage1_petsc_backend PRIVATE cpp/include)
target_link_libraries(stage1_petsc_backend PRIVATE PkgConfig::PETSC MPI::MPI_CXX)
target_compile_definitions(stage1_petsc_backend PRIVATE
  PETSC_FUNCTION_NAME_CXX=__func__
  PETSC_CXX_RESTRICT=__restrict__
)

add_executable(stage2_dynamic_forward
  cpp/src/stage2/main_stage2.cpp
  cpp/src/stage2/MatlabLikeData.cpp
  cpp/src/stage2/FEMAssembly.cpp
  cpp/src/stage2/PetscEigenBridge.cpp
  cpp/src/stage2/DynamicElasticLinear.cpp
  cpp/src/stage2/Optimization.cpp
  cpp/src/ROM/Utils/rom_petsc_utils.cpp
  cpp/src/ROM/Error_Estimator/RelativeResidualEstimator.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/UpdateBase.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/MyPOD_SVD1.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/IncrementalSVD.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/MyIncrementalQR1.cpp
  cpp/src/ROM/ROM_method/ROM_Base.cpp
  cpp/src/ROM/ROM_method/FOM_direct.cpp
  cpp/src/ROM/ROM_method/FOM_iterative.cpp
  cpp/src/ROM/ROM_method/MyStrategy.cpp
  cpp/src/ROM/ROM_method/ROM_factory.cpp
  cpp/third_party/mma/MMASolver.cpp
  cpp/third_party/mma_matlab_port/MatlabMMA.cpp
  cpp/third_party/mma_jdumas/JdumasMMASolver.cpp
)

add_executable(basis_update_benchmark
  cpp/src/tools/basis_update_benchmark.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/UpdateBase.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/MyPOD_SVD1.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/IncrementalSVD.cpp
  cpp/src/ROM/Reduced_Basis_Agorithm/MyIncrementalQR1.cpp
)

target_include_directories(stage2_dynamic_forward PRIVATE cpp/include)
target_include_directories(basis_update_benchmark PRIVATE cpp/include)
target_include_directories(stage2_dynamic_forward SYSTEM PRIVATE
  cpp/third_party/eigen
  cpp/third_party/mma
  cpp/third_party/mma_matlab_port
  cpp/third_party/mma_jdumas
)
target_include_directories(basis_update_benchmark SYSTEM PRIVATE
  cpp/third_party/eigen
)
target_link_libraries(stage2_dynamic_forward PRIVATE PkgConfig::PETSC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_link_libraries(basis_update_benchmark PRIVATE PkgConfig::PETSC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_definitions(stage2_dynamic_forward PRIVATE
  PETSC_FUNCTION_NAME_CXX=__func__
  PETSC_CXX_RESTRICT=__restrict__
)
target_compile_definitions(basis_update_benchmark PRIVATE
  PETSC_FUNCTION_NAME_CXX=__func__
  PETSC_CXX_RESTRICT=__restrict__
)

set(STAGE2_MKLROOT "$ENV{MKLROOT}" CACHE PATH "Intel oneAPI MKL root")
if(NOT STAGE2_MKLROOT AND EXISTS "/opt/intel/oneapi/mkl/2025.0/include/mkl_lapacke.h")
  set(STAGE2_MKLROOT "/opt/intel/oneapi/mkl/2025.0")
endif()

if(STAGE2_MKLROOT AND EXISTS "${STAGE2_MKLROOT}/include/mkl_lapacke.h")
  target_include_directories(stage2_dynamic_forward SYSTEM PRIVATE "${STAGE2_MKLROOT}/include")
  target_include_directories(basis_update_benchmark SYSTEM PRIVATE "${STAGE2_MKLROOT}/include")
  target_link_directories(stage2_dynamic_forward PRIVATE "${STAGE2_MKLROOT}/lib" "${STAGE2_MKLROOT}/lib/intel64")
  target_link_directories(basis_update_benchmark PRIVATE "${STAGE2_MKLROOT}/lib" "${STAGE2_MKLROOT}/lib/intel64")
  target_link_libraries(stage2_dynamic_forward PRIVATE "${STAGE2_MKLROOT}/lib/libmkl_rt.so" dl pthread m)
  target_link_libraries(basis_update_benchmark PRIVATE "${STAGE2_MKLROOT}/lib/libmkl_rt.so" dl pthread m)
  target_compile_definitions(stage2_dynamic_forward PRIVATE EIGEN_USE_LAPACKE)
  target_compile_definitions(basis_update_benchmark PRIVATE EIGEN_USE_LAPACKE)
endif()

if (MSVC)
  target_compile_options(stage1_petsc_backend PRIVATE /W4)
  target_compile_options(stage2_dynamic_forward PRIVATE /W4)
else()
  target_compile_options(stage1_petsc_backend PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(stage2_dynamic_forward PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(basis_update_benchmark PRIVATE -Wall -Wextra -Wpedantic)
  set_source_files_properties(
    cpp/src/ROM/Reduced_Basis_Agorithm/MyPOD_SVD1.cpp
    PROPERTIES COMPILE_OPTIONS -Wno-deprecated-declarations
  )
endif()
