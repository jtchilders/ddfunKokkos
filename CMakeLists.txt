# define project using C++
cmake_minimum_required(VERSION 3.10)
project(ddfunKokkos VERSION 1.0.0 LANGUAGES CXX Fortran)

# Enable compile commands generation for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# depends on Kokkos
find_package(Kokkos REQUIRED)

# add directories
add_subdirectory(src)
if(BUILD_UNIT_TESTS)
   # print confirmation
   message(STATUS "BUILDING UNIT TESTS:   ENABLED")
   add_subdirectory(unitTests)
else()
   message(STATUS "BUILDING UNIT TESTS:   DISABLED")
endif()

# Create installable target
add_library(ddfunKokkos INTERFACE)
target_include_directories(ddfunKokkos INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                             $<INSTALL_INTERFACE:include>)
target_link_libraries(ddfunKokkos INTERFACE Kokkos::Kokkos)

# Install headers
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")

# Install target
install(TARGETS ddfunKokkos EXPORT ddfunKokkosTargets)
install(EXPORT ddfunKokkosTargets
        FILE ddfunKokkosConfig.cmake
        NAMESPACE ddfunKokkos::
        DESTINATION lib/cmake/ddfunKokkos)

# Create package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ddfunKokkosConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ddfunKokkosConfigVersion.cmake
    DESTINATION lib/cmake/ddfunKokkos
)
