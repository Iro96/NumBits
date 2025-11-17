cmake_minimum_required(VERSION 3.15)
project(NumBits VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(NUMBITS_BUILD_TESTS "Build NumBits tests" ON)
option(NUMBITS_BUILD_EXAMPLES "Build NumBits examples" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(NUMBITS_SOURCES
    src/array.cpp
    src/operations.cpp
    src/math_functions.cpp
    src/linear_algebra.cpp
    src/broadcasting.cpp
    src/array_manipulation.cpp
    src/indexing.cpp
)

set(NUMBITS_HEADERS
    include/numbits/array.hpp
    include/numbits/operations.hpp
    include/numbits/math_functions.hpp
    include/numbits/linear_algebra.hpp
    include/numbits/broadcasting.hpp
    include/numbits/array_manipulation.hpp
    include/numbits/indexing.hpp
    include/numbits/io.hpp
    include/numbits/types.hpp
    include/numbits/utils.hpp
    include/numbits/numbits.hpp
)

# Create library
add_library(numbits STATIC ${NUMBITS_SOURCES} ${NUMBITS_HEADERS})

target_include_directories(numbits PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Build examples
if(NUMBITS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(NUMBITS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS numbits
    EXPORT NumBitsTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/numbits
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT NumBitsTargets
    FILE NumBitsTargets.cmake
    NAMESPACE NumBits::
    DESTINATION lib/cmake/NumBits
)

