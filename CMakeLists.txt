cmake_minimum_required(VERSION 3.14)
project(NumBits LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMBITS_BUILD_TESTS "Build NumBits tests" ON)
option(NUMBITS_BUILD_EXAMPLES "Build NumBits examples" ON)

add_library(numbits INTERFACE)
target_include_directories(numbits INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Only enable tests/examples if directories exist
if(NUMBITS_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()

if(NUMBITS_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()
