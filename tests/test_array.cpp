#include <iostream>
#include <cassert>
#include "numbits/numbits.hpp"

using namespace numbits;

#define TEST_CASE(name) void name()
#define RUN_TEST(name)  \
    std::cout << "Running " #name "... "; \
    name(); \
    std::cout << "OK\n";

TEST_CASE(test_ndarray_creation) {
    ndarray<float> arr({2, 3});
    assert((arr.shape() == Shape{2, 3}));
    assert(arr.size() == 6);
    assert(arr.ndim() == 2);
}

TEST_CASE(test_ndarray_with_data) {
    ndarray<float> arr({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    assert(arr[0] == 1.0f);
    assert(arr[1] == 2.0f);
    assert(arr[2] == 3.0f);
    assert(arr[3] == 4.0f);
}

TEST_CASE(test_ndarray_zeros_ones) {
    auto zeros = ndarray<float>::zeros({3, 3});
    assert(zeros[0] == 0.0f);
    assert(zeros.size() == 9);

    auto ones = ndarray<float>::ones({2, 2});
    assert(ones[0] == 1.0f);
    assert(ones.size() == 4);
}

TEST_CASE(test_ndarray_reshape) {
    ndarray<float> arr({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto reshaped = arr.reshape({3, 2});
    assert((reshaped.shape() == Shape{3, 2}));
    assert(reshaped.size() == 6);
}

TEST_CASE(test_ndarray_element_access) {
    ndarray<float> arr({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    assert(arr.at({0, 0}) == 1.0f);
    assert(arr.at({0, 1}) == 2.0f);
    assert(arr.at({1, 0}) == 3.0f);
    assert(arr.at({1, 1}) == 4.0f);
}

int main() {
    RUN_TEST(test_ndarray_creation);
    RUN_TEST(test_ndarray_with_data);
    RUN_TEST(test_ndarray_zeros_ones);
    RUN_TEST(test_ndarray_reshape);
    RUN_TEST(test_ndarray_element_access);

    std::cout << "All tests passed!\n";
    return 0;
}
