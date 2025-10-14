add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE numbits)
add_test(NAME test_core COMMAND test_core)

add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE numbits)
add_test(NAME test_linalg COMMAND test_linalg)