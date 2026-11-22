cmake_minimum_required(VERSION 3.20)
project(tilesim CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -ffp-contract=off)

find_package(nlohmann_json CONFIG REQUIRED)

add_library(core machine.cpp program.cpp gemm.cpp sim.cpp out.cpp)
target_link_libraries(core PUBLIC nlohmann_json::nlohmann_json)

add_executable(tilesim main.cpp)
target_link_libraries(tilesim PRIVATE core)