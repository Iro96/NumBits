add_executable(example_basic example_basic.cpp)
target_link_libraries(example_basic PRIVATE numbits)

add_executable(example_linalg example_linalg.cpp)
target_link_libraries(example_linalg PRIVATE numbits)

add_executable(example_math example_math.cpp)
target_link_libraries(example_math PRIVATE numbits)

add_executable(example_v05 example_v05.cpp)
target_link_libraries(example_v05 PRIVATE numbits)
