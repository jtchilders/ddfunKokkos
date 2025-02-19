
include(FetchContent)

# Fetch Catch2 (version 2.x or 3.x; adjust GIT_TAG as needed)
FetchContent_Declare(
   catch2
   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
   GIT_TAG        v3.8.0
)
FetchContent_MakeAvailable(catch2)

add_executable(unit_tests unit_tests.cpp ddfuna_partial.f90 ${DDFUN_KOKKOS_SRC})
target_link_libraries(unit_tests PRIVATE Kokkos::kokkos Catch2::Catch2)
target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
