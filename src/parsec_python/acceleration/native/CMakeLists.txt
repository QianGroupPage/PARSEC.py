cmake_minimum_required(VERSION 3.24)

project(parsec_accelerated_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(
    parsec_accelerated_native
    MODULE
    module.cpp
    ca_lda.cpp
    conjugate_gradient.cpp
    finite_difference.cpp
    fused_hamiltonian.cpp
    hartree_boundary.cpp
    radial_grid.cpp
    sector_operator.cpp
    sector_stencil.cpp
)

target_link_libraries(parsec_accelerated_native PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(
    parsec_accelerated_native
    PRIVATE
    PYBIND11_DETAILED_ERROR_MESSAGES=1
)

# Keep IEEE-style arithmetic.  In particular, fast-math may reassociate the
# short stencil/projector sums and move eigensolver convergence decisions.
if(MSVC)
    target_compile_options(
        parsec_accelerated_native
        PRIVATE
        /O2
        /permissive-
        /EHsc
        /fp:precise
    )
else()
    target_compile_options(
        parsec_accelerated_native
        PRIVATE
        -O3
        -fno-fast-math
    )
endif()

install(
    TARGETS parsec_accelerated_native
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
