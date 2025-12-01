/**
 * @file test_operations.cpp
 * @brief Unit tests for array operations (element-wise, reductions, broadcasting).
 *
 * Tests the following:
 *   - Basic arithmetic operations (addition, subtraction, multiplication, division)
 *   - Scalar operations
 *   - Reduction operations (sum, mean, min, max)
 *   - Broadcasting with where, clip
 *   - Logical operations (and, or, xor, not)
 *   - Boolean reductions (all, any)
 *   - Cumulative operations (cumsum, cumprod)
 *   - Index finding (argmax, argmin)
 *
 * @date 2025
 */

#include <iostream>
#include <cassert>
#include "numbits/numbits.hpp"

using namespace numbits;

#define TEST_CASE(name) void name()
#define RUN_TEST(name)  \
    std::cout << "Running " #name "... "; \
    name(); \
    std::cout << "OK\n";

/**
 * @brief Test element-wise addition of two arrays.
 */
TEST_CASE(test_addition) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    auto c = a + b;
    assert(c[0] == 6.0f);
    assert(c[1] == 8.0f);
    assert(c[2] == 10.0f);
    assert(c[3] == 12.0f);
}

/**
 * @brief Test addition of scalar to array.
 */
TEST_CASE(test_scalar_addition) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = a + 5.0f;
    assert(b[0] == 6.0f);
    assert(b[1] == 7.0f);
    assert(b[2] == 8.0f);
    assert(b[3] == 9.0f);
}

/**
 * @brief Test element-wise multiplication.
 */
TEST_CASE(test_multiplication) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> b({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    auto c = a * b;
    assert(c[0] == 2.0f);
    assert(c[1] == 4.0f);
    assert(c[2] == 6.0f);
    assert(c[3] == 8.0f);
}

/**
 * @brief Test sum reduction operation.
 */
TEST_CASE(test_sum_reduction) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    assert(sum(a) == 10.0f);
}

/**
 * @brief Test mean reduction operation.
 */
TEST_CASE(test_mean_reduction) {
    ndarray<float> a({2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
    assert(mean(a) == 5.0f);
}

/**
 * @brief Test where operation with broadcasting.
 */
TEST_CASE(test_where_broadcasting) {
    ndarray<bool> condition({2, 1}, {true, false});
    ndarray<float> x({1, 3}, {1.0f, 2.0f, 3.0f});
    ndarray<float> y({}, {0.0f});

    auto result = where(condition, x, y);
    assert((result.shape() == Shape{2, 3}));
    assert(result[0] == 1.0f);
    assert(result[1] == 2.0f);
    assert(result[2] == 3.0f);
    assert(result[3] == 0.0f);
    assert(result[4] == 0.0f);
    assert(result[5] == 0.0f);
}

/**
 * @brief Test clipping values to a range (scalar bounds).
 */
TEST_CASE(test_clip_scalar) {
    ndarray<float> values({4}, {-1.0f, 0.25f, 0.75f, 2.0f});
    auto clipped = clip(values, 0.0f, 1.0f);
    assert(clipped[0] == 0.0f);
    assert(clipped[1] == 0.25f);
    assert(clipped[2] == 0.75f);
    assert(clipped[3] == 1.0f);
}

/**
 * @brief Test clipping with broadcasted array bounds.
 */
TEST_CASE(test_clip_broadcast) {
    ndarray<float> values({2, 2}, {-1.0f, 0.2f, 1.2f, 0.4f});
    ndarray<float> mins({1, 2}, {0.0f, 0.1f});
    ndarray<float> maxs({2, 1}, {0.5f, 0.9f});
    auto clipped = clip(values, mins, maxs);
    assert((clipped.shape() == Shape{2, 2}));
    assert(clipped[0] == 0.0f);
    assert(clipped[1] == 0.2f);
    assert(clipped[2] == 0.5f);
    assert(clipped[3] == 0.4f);
}

/**
 * @brief Test finding index of maximum and minimum values.
 */
TEST_CASE(test_argmax_argmin) {
    ndarray<int> values({5}, {3, 1, 7, 7, -2});
    auto max_index = argmax(values);
    auto min_index = argmin(values);
    assert(max_index == 2 || max_index == 3);
    assert(min_index == 4);
}

/**
 * @brief Test logical operations (and, or, xor, not).
 */
TEST_CASE(test_logical_operations) {
    ndarray<int> lhs({2, 2}, {0, 1, 2, 0});
    ndarray<int> rhs({1, 2}, {0, 1});

    auto land = logical_and(lhs, rhs);
    auto lor = logical_or(lhs, rhs);
    auto lxor = logical_xor(lhs, rhs);
    auto lnot = logical_not(lhs);

    assert((land.shape() == Shape{2, 2}));
    assert(land[0] == false);
    assert(land[1] == true);
    assert(land[2] == false);
    assert(land[3] == false);

    assert(lor[0] == false);
    assert(lor[1] == true);
    assert(lor[2] == true);
    assert(lor[3] == true);

    assert(lxor[0] == false);
    assert(lxor[1] == false);
    assert(lxor[2] == true);
    assert(lxor[3] == true);

    assert(lnot[0] == true);
    assert(lnot[1] == false);
    assert(lnot[2] == false);
    assert(lnot[3] == true);
}

/**
 * @brief Test boolean reductions (all, any).
 */
TEST_CASE(test_all_any) {
    ndarray<int> values({4}, {1, 2, 0, 3});
    assert(all(values) == false);
    assert(any(values) == true);

    ndarray<int> zeros({3}, {0, 0, 0});
    assert(all(zeros) == false);
    assert(any(zeros) == false);
}

/**
 * @brief Test cumulative operations (cumsum, cumprod).
 */
TEST_CASE(test_cumulative_operations) {
    ndarray<int> values({5}, {1, 2, 3, 4, 5});
    auto sums = cumsum(values);
    auto prods = cumprod(values);

    assert((sums[0] == 1 && sums[1] == 3 && sums[2] == 6 && sums[3] == 10 && sums[4] == 15));
    assert((prods[0] == 1 && prods[1] == 2 && prods[2] == 6 && prods[3] == 24 && prods[4] == 120));
}

/**
 * @brief Test subtraction of arrays.
 */
TEST_CASE(test_subtraction) {
    ndarray<float> a({2, 2}, {10.0f, 8.0f, 6.0f, 4.0f});
    ndarray<float> b({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto c = a - b;
    assert(c[0] == 9.0f);
    assert(c[1] == 6.0f);
    assert(c[2] == 3.0f);
    assert(c[3] == 0.0f);
}

/**
 * @brief Test division of arrays.
 */
TEST_CASE(test_division) {
    ndarray<float> a({2, 2}, {10.0f, 8.0f, 6.0f, 4.0f});
    ndarray<float> b({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    auto c = a / b;
    assert(c[0] == 5.0f);
    assert(c[1] == 4.0f);
    assert(c[2] == 3.0f);
    assert(c[3] == 2.0f);
}

/**
 * @brief Test min and max reduction operations.
 */
TEST_CASE(test_min_max_reduction) {
    ndarray<int> a({3}, {5, 2, 9});
    assert(min(a) == 2);
    assert(max(a) == 9);
}

/**
 * @brief Test scalar multiplication.
 */
TEST_CASE(test_scalar_multiplication) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = a * 3.0f;
    assert(b[0] == 3.0f);
    assert(b[1] == 6.0f);
    assert(b[2] == 9.0f);
    assert(b[3] == 12.0f);
}

int main() {
    RUN_TEST(test_addition);
    RUN_TEST(test_scalar_addition);
    RUN_TEST(test_multiplication);
    RUN_TEST(test_sum_reduction);
    RUN_TEST(test_mean_reduction);
    RUN_TEST(test_where_broadcasting);
    RUN_TEST(test_clip_scalar);
    RUN_TEST(test_clip_broadcast);
    RUN_TEST(test_argmax_argmin);
    RUN_TEST(test_logical_operations);
    RUN_TEST(test_all_any);
    RUN_TEST(test_cumulative_operations);
    RUN_TEST(test_subtraction);
    RUN_TEST(test_division);
    RUN_TEST(test_min_max_reduction);
    RUN_TEST(test_scalar_multiplication);

    std::cout << "All tests passed!\n";
    return 0;
}
