/**
 * @file test_array.cpp
 * @brief Unit tests for array creation and basic ndarray functionality.
 *
 * Tests the following:
 *   - Array creation and shape verification
 *   - Array initialization with data
 *   - Factory methods (zeros, ones)
 *   - Reshape and flatten operations
 *   - Element access methods
 *   - Array creation functions (arange, linspace, eye)
 *
 * @date 2025
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include "numbits/numbits.hpp"

using namespace numbits;

#define TEST_CASE(name) void name()
#define RUN_TEST(name)  \
    std::cout << "Running " #name "... "; \
    name(); \
    std::cout << "OK\n";

/**
 * @brief Test basic ndarray creation with specified shape.
 */
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

TEST_CASE(test_arange_creation) {
    auto seq = arange<int>(5);
    assert(seq.size() == 5);
    for (int i = 0; i < 5; ++i) {
        assert(seq[i] == i);
    }

    auto odd = arange<int>(1, 6, 2);
    assert((odd.shape() == Shape{3}));
    assert((odd[0] == 1 && odd[1] == 3 && odd[2] == 5));
}

TEST_CASE(test_linspace_creation) {
    auto seq = linspace<double>(0.0, 1.0, 5);
    assert(seq.size() == 5);
    assert(std::abs(seq[0] - 0.0) < 1e-9);
    assert(std::abs(seq[2] - 0.5) < 1e-9);
    assert(std::abs(seq[4] - 1.0) < 1e-9);

    auto open = linspace<double>(0.0, 1.0, 4, false);
    assert(std::abs(open[3] - 0.75) < 1e-9);
}

TEST_CASE(test_eye_creation) {
    auto identity = eye<float>(3);
    assert((identity.shape() == Shape{3, 3}));
    assert(identity[0] == 1.0f);
    assert(identity[1] == 0.0f);
    assert(identity[3] == 0.0f);
    assert(identity[4] == 1.0f);

    auto offset = eye<float>(2, 4, 1);
    assert((offset.shape() == Shape{2, 4}));
    assert(offset[0] == 0.0f);
    assert(offset[1] == 1.0f);
    assert(offset[4] == 0.0f);
    assert(offset[5] == 0.0f);
}

/**
 * @brief Test copy construction and assignment operators.
 */
TEST_CASE(test_ndarray_copy) {
    ndarray<float> original({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> copied(original);
    assert(copied.shape() == original.shape());
    assert(copied[0] == original[0]);
    assert(copied[3] == original[3]);
}

/**
 * @brief Test move semantics for efficient array transfers.
 */
TEST_CASE(test_ndarray_move) {
    ndarray<float> original({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> moved = std::move(original);
    assert((moved.shape() == Shape{2, 2}));
    assert(moved[0] == 1.0f);
}

/**
 * @brief Test full array creation with a specific value.
 */
TEST_CASE(test_ndarray_full) {
    ndarray<float> filled = ndarray<float>::full({2, 3}, 5.5f);
    assert((filled.shape() == Shape{2, 3}));
    for (size_t i = 0; i < filled.size(); ++i) {
        assert(filled[i] == 5.5f);
    }
}

/**
 * @brief Test flatten operation preserves all data correctly.
 */
TEST_CASE(test_ndarray_flatten) {
    ndarray<int> arr({2, 3}, {1, 2, 3, 4, 5, 6});
    auto flat = arr.flatten();
    assert(flat.size() == 6);
    assert((flat.shape() == Shape{6}));
    for (int i = 0; i < 6; ++i) {
        assert(flat[i] == i + 1);
    }
}

/**
 * @brief Test ndim() returns correct number of dimensions.
 */
TEST_CASE(test_ndarray_ndim) {
    ndarray<float> arr1d({5});
    ndarray<float> arr2d({2, 3});
    ndarray<float> arr3d({2, 3, 4});
    assert(arr1d.ndim() == 1);
    assert(arr2d.ndim() == 2);
    assert(arr3d.ndim() == 3);
}

/**
 * @brief Test iterators for range-based loops and algorithms.
 */
TEST_CASE(test_ndarray_iterators) {
    ndarray<int> arr({2, 2}, {1, 2, 3, 4});
    int sum = 0;
    for (auto val : arr) {
        sum += val;
    }
    assert(sum == 10);
}

int main() {
    RUN_TEST(test_ndarray_creation);
    RUN_TEST(test_ndarray_with_data);
    RUN_TEST(test_ndarray_zeros_ones);
    RUN_TEST(test_ndarray_reshape);
    RUN_TEST(test_ndarray_element_access);
    RUN_TEST(test_arange_creation);
    RUN_TEST(test_linspace_creation);
    RUN_TEST(test_eye_creation);
    RUN_TEST(test_ndarray_copy);
    RUN_TEST(test_ndarray_move);
    RUN_TEST(test_ndarray_full);
    RUN_TEST(test_ndarray_flatten);
    RUN_TEST(test_ndarray_ndim);
    RUN_TEST(test_ndarray_iterators);

    std::cout << "All tests passed!\n";
    return 0;
}
