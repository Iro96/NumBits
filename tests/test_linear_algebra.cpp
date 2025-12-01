/**
 * @file test_linear_algebra.cpp
 * @brief Unit tests for linear algebra operations.
 *
 * Tests the following:
 *   - Matrix multiplication (matmul)
 *   - Matrix transpose
 *   - Determinant calculation (2x2 matrices)
 *   - Matrix inverse
 *   - Matrix trace (sum of diagonal elements)
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
 * @brief Test basic matrix multiplication (matmul).
 */
TEST_CASE(test_matrix_multiplication) {
    ndarray<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ndarray<float> b({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto c = matmul(a, b);
    assert((c.shape() == Shape{2, 2}));
    assert(c.at({0, 0}) == 22.0f);
    assert(c.at({0, 1}) == 28.0f);
    assert(c.at({1, 0}) == 49.0f);
    assert(c.at({1, 1}) == 64.0f);
}

/**
 * @brief Test matrix transpose operation.
 */
TEST_CASE(test_transpose) {
    ndarray<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto at = transpose(a);
    assert((at.shape() == Shape{3, 2}));
    assert(at.at({0, 0}) == 1.0f);
    assert(at.at({0, 1}) == 4.0f);
    assert(at.at({1, 0}) == 2.0f);
    assert(at.at({1, 1}) == 5.0f);
}

/**
 * @brief Test determinant calculation for 2x2 matrices.
 */
TEST_CASE(test_determinant_2x2) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    assert(determinant(a) == -2.0f);
}

/**
 * @brief Test matrix inverse computation and verification.
 */
TEST_CASE(test_inverse_2x2) {
    ndarray<float> a({2, 2}, {4.0f, 7.0f, 2.0f, 6.0f});
    auto inv = inverse(a);
    auto identity = matmul(a, inv);
    assert(std::abs(identity.at({0, 0}) - 1.0f) < 1e-5f);
    assert(std::abs(identity.at({1, 1}) - 1.0f) < 1e-5f);
}

/**
 * @brief Test matrix trace (sum of diagonal elements).
 */
TEST_CASE(test_trace) {
    ndarray<float> a({3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });
    assert(trace(a) == 15.0f);
}

/**
 * @brief Test 3x3 matrix multiplication.
 */
TEST_CASE(test_matrix_multiplication_3x3) {
    ndarray<float> a({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    ndarray<float> b({3, 3}, {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f});
    auto c = matmul(a, b);
    assert((c.shape() == Shape{3, 3}));
    assert(c.at({0, 0}) == 30.0f);
}

/**
 * @brief Test chained matrix multiplications.
 */
TEST_CASE(test_chained_matmul) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> b({2, 2}, {2.0f, 0.0f, 1.0f, 2.0f});
    auto ab = matmul(a, b);
    auto abb = matmul(ab, b);
    assert((abb.shape() == Shape{2, 2}));
}

/**
 * @brief Test transpose of transpose returns original shape.
 */
TEST_CASE(test_transpose_twice) {
    ndarray<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto at = transpose(a);
    auto att = transpose(at);
    assert((att.shape() == a.shape()));
    for (size_t i = 0; i < a.size(); ++i) {
        assert(att[i] == a[i]);
    }
}

/**
 * @brief Test square matrix identity: trace = sum of eigenvalues for simple cases.
 */
TEST_CASE(test_trace_diagonal_matrix) {
    ndarray<float> diag({3, 3}, {
        5.0f, 0.0f, 0.0f,
        0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 2.0f
    });
    assert(trace(diag) == 10.0f);
}

int main() {
    RUN_TEST(test_matrix_multiplication);
    RUN_TEST(test_transpose);
    RUN_TEST(test_determinant_2x2);
    RUN_TEST(test_inverse_2x2);
    RUN_TEST(test_trace);
    RUN_TEST(test_matrix_multiplication_3x3);
    RUN_TEST(test_chained_matmul);
    RUN_TEST(test_transpose_twice);
    RUN_TEST(test_trace_diagonal_matrix);

    std::cout << "All tests passed!\n";
    return 0;
}
