cmake_minimum_required(VERSION 3.14)
project(NumBits VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_library(numbits INTERFACE)
target_include_directories(numbits INTERFACE include)

add_executable(example examples/example_basic.cpp)
target_link_libraries(example PRIVATE numbits)
