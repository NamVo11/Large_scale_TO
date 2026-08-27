cmake_minimum_required(VERSION 3.18)
project(Large_scale_TO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED CXX)
find_package(OpenMP REQUIRED)
set(PETSC_PKG "PETSc" CACHE STRING "PETSc pkg-config package, e.g. petsc-dmo on MSYS2 UCRT64")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET ${PETSC_PKG})

add_executable(stage2_dynamic_forward
  src/main_stage2.cpp
  src/Analysis/Analysis.cpp
  src/BounadryCondition/Function.cpp
  src/BounadryCondition/BoundaryCondition.cpp
  src/BounadryCondition/RangeBoundaryCondition.cpp
  src/BounadryCondition/HMSetBoundaryCondition.cpp
  src/FEM/FEM.cpp
  src/FEM/Element/ElementBase.cpp
  src/FEM/Element/Q4Element.cpp
  src/FEM/Element/Tri3Element.cpp
  src/FEM/Element/H8Element.cpp
  src/FEM/Element/T4Element.cpp
  src/FEM/Element/P6Element.cpp
  src/Input/Initialization.cpp
  src/Region/Region.cpp
  src/Optimization/Interpolation/RAM.cpp
  src/Optimization/Interpolation/SIMP.cpp
  src/Optimization/Interpolation/DMO.cpp
  src/Optimization/Filtering/Filtering.cpp
  src/Optimization/Optimization.cpp
  src/Optimization/Volume/VolumeFunctional.cpp
  src/Optimization/Volume/VolumeFractionConstraint.cpp
  src/Optimization/SensitivityMethod/AVM.cpp
  src/Optimization/SensitivityMethod/Debug.cpp
  src/Output/OutputData.cpp
  src/helper/MyAlgebra.cpp
  src/helper/PetscEigenBridge.cpp
  src/helper/PetscEigenHelper.cpp
  src/Analysis/DynamicLinearElastic/PetscDynamicLinearElastic.cpp
  src/Analysis/DynamicLinearElastic/EigenDynamicLinearElastic.cpp
  src/Analysis/StaticLinearElastic/PetscStaticLinearElastic.cpp
  src/Analysis/StaticLinearElastic/EigenStaticLinearElastic.cpp
  src/Analysis/SensitivityBase.cpp
  src/Analysis/DynamicLinearElastic/DynamicLinearElasticFunctional.cpp
  src/Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.cpp
  src/Analysis/DynamicLinearElastic/EigenDynamicLinearElasticAVM.cpp
  src/Analysis/DynamicLinearElastic/PetscDynamicLinearElasticAVM.cpp
  src/Analysis/StaticLinearElastic/StaticLinearElasticFunctional.cpp
  src/Analysis/StaticLinearElastic/StaticLinearElasticAVM.cpp
  src/Analysis/StaticLinearElastic/EigenStaticLinearElasticAVM.cpp
  src/ROM/ROM_method/ROM_Base.cpp
  src/ROM/ROM_method/FOM_direct.cpp
  src/ROM/ROM_method/FOM_iterative.cpp
  src/ROM/ROM_method/PetscFOM_direct.cpp
  src/ROM/ROM_method/PetscFOM_iterative.cpp
  src/ROM/ROM_method/MyStrategy.cpp
  src/ROM/ROM_method/InitilizeROM.cpp
  src/ROM/Error_Estimator/RelativeResidualEstimator.cpp
  src/ROM/Reduced_Basis_Algorithm/UpdateBase.cpp
  src/ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.cpp
  src/ROM/Reduced_Basis_Algorithm/IncrementalSVD.cpp
  src/ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.cpp
  src/third_party/mma/MMASolver.cpp
  src/Optimization/Algorithm/MatlabMMA.cpp
  src/third_party/mma_jdumas/JdumasMMASolver.cpp
)

add_executable(basis_update_benchmark
  src/tools/basis_update_benchmark.cpp
  src/helper/MyAlgebra.cpp
  src/ROM/Reduced_Basis_Algorithm/UpdateBase.cpp
  src/ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.cpp
  src/ROM/Reduced_Basis_Algorithm/IncrementalSVD.cpp
  src/ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.cpp
)

target_include_directories(stage2_dynamic_forward PRIVATE src)
target_include_directories(basis_update_benchmark PRIVATE src)
target_include_directories(stage2_dynamic_forward SYSTEM PRIVATE
  src/third_party/eigen
  src/third_party/json
  src/third_party/mma
  src/third_party/mma_jdumas
)
target_include_directories(basis_update_benchmark SYSTEM PRIVATE
  src/third_party/eigen
  src/third_party/json
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
option(STAGE2_EIGEN_USE_LAPACKE "Use Eigen LAPACKE/PardisoLDLT path; affects only LINEAR_BACKEND=eigen" ON)
option(STAGE2_EIGEN_USE_MKL_BLAS "Use Intel MKL BLAS/LAPACK backend for Eigen dense operations" OFF)
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
  if(STAGE2_EIGEN_USE_LAPACKE)
    target_compile_definitions(stage2_dynamic_forward PRIVATE EIGEN_USE_LAPACKE)
    target_compile_definitions(basis_update_benchmark PRIVATE EIGEN_USE_LAPACKE)
  endif()
  if(STAGE2_EIGEN_USE_MKL_BLAS)
    target_compile_definitions(stage2_dynamic_forward PRIVATE EIGEN_USE_BLAS)
    target_compile_definitions(basis_update_benchmark PRIVATE EIGEN_USE_BLAS)
  endif()
endif()

if (MSVC)
  target_compile_options(stage2_dynamic_forward PRIVATE /W4)
else()
  target_compile_options(stage2_dynamic_forward PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(basis_update_benchmark PRIVATE -Wall -Wextra -Wpedantic)
  set_source_files_properties(
    src/ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.cpp
    PROPERTIES COMPILE_OPTIONS -Wno-deprecated-declarations
  )
endif()
