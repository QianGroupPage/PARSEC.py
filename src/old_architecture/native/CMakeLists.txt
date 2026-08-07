cmake_minimum_required(VERSION 3.24)

project(rsdft_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(
    rsdft_native
    MODULE
    module.cpp
    pseudo_diag_omp.cpp
    pseudo_nl_omp.cpp
)

target_link_libraries(rsdft_native PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(rsdft_native PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(MSVC)
    target_compile_options(rsdft_native PRIVATE /permissive- /EHsc)
else()
    target_compile_options(rsdft_native PRIVATE -O3)
endif()

install(TARGETS rsdft_native LIBRARY DESTINATION . RUNTIME DESTINATION .)
