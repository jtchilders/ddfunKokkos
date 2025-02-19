# define project using C++
cmake_minimum_required(VERSION 3.10)
project(KOKKOS_DDFUN LANGUAGES CXX Fortran)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# depends on Kokkos
find_package(Kokkos REQUIRED)

# add directories
add_subdirectory(src)
add_subdirectory(unitTests)
