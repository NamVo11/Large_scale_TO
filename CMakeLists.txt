cmake_minimum_required(VERSION 3.18)
project(stage1_petsc_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED CXX)
set(PETSC_PKG "PETSc" CACHE STRING "PETSc pkg-config package, e.g. petsc-dmo on MSYS2 UCRT64")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET ${PETSC_PKG})

add_executable(stage1_petsc_backend
  cpp/src/main.cpp
  cpp/src/IO.cpp
  cpp/src/FEMH8.cpp
  cpp/src/DynamicElasticSolver.cpp
)

target_include_directories(stage1_petsc_backend PRIVATE cpp/include)
target_link_libraries(stage1_petsc_backend PRIVATE PkgConfig::PETSC MPI::MPI_CXX)
target_compile_definitions(stage1_petsc_backend PRIVATE
  PETSC_FUNCTION_NAME_CXX=__func__
  PETSC_CXX_RESTRICT=__restrict__
)

if (MSVC)
  target_compile_options(stage1_petsc_backend PRIVATE /W4)
else()
  target_compile_options(stage1_petsc_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()
