cmake_minimum_required(VERSION 3.15)

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found. Tests will not be built.")
    return()
endif()

# Test executables
add_executable(test_array test_array.cpp)
target_link_libraries(test_array numbits Catch2::Catch2)

add_executable(test_operations test_operations.cpp)
target_link_libraries(test_operations numbits Catch2::Catch2)

add_executable(test_linear_algebra test_linear_algebra.cpp)
target_link_libraries(test_linear_algebra numbits Catch2::Catch2)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io numbits Catch2::Catch2)

# Register tests
add_test(NAME ArrayTests COMMAND test_array)
add_test(NAME OperationsTests COMMAND test_operations)
add_test(NAME LinearAlgebraTests COMMAND test_linear_algebra)
add_test(NAME IOTests COMMAND test_io)
