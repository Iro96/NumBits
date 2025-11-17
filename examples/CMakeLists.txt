cmake_minimum_required(VERSION 3.15)

# Example 1: Basic array operations
add_executable(example_basic example_basic.cpp)
target_link_libraries(example_basic numbits)

# Example 2: Linear algebra
add_executable(example_linear_algebra example_linear_algebra.cpp)
target_link_libraries(example_linear_algebra numbits)

# Example 3: Math functions
add_executable(example_math example_math.cpp)
target_link_libraries(example_math numbits)

# Example 4: Broadcasting
add_executable(example_broadcasting example_broadcasting.cpp)
target_link_libraries(example_broadcasting numbits)
