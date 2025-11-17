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

TEST_CASE(test_transpose) {
    ndarray<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto at = transpose(a);
    assert((at.shape() == Shape{3, 2}));
    assert(at.at({0, 0}) == 1.0f);
    assert(at.at({0, 1}) == 4.0f);
    assert(at.at({1, 0}) == 2.0f);
    assert(at.at({1, 1}) == 5.0f);
}

TEST_CASE(test_determinant_2x2) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    assert(determinant(a) == -2.0f);
}

TEST_CASE(test_inverse_2x2) {
    ndarray<float> a({2, 2}, {4.0f, 7.0f, 2.0f, 6.0f});
    auto inv = inverse(a);
    auto identity = matmul(a, inv);
    assert(std::abs(identity.at({0, 0}) - 1.0f) < 1e-5f);
    assert(std::abs(identity.at({1, 1}) - 1.0f) < 1e-5f);
}

TEST_CASE(test_trace) {
    ndarray<float> a({3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });
    assert(trace(a) == 15.0f);
}

int main() {
    RUN_TEST(test_matrix_multiplication);
    RUN_TEST(test_transpose);
    RUN_TEST(test_determinant_2x2);
    RUN_TEST(test_inverse_2x2);
    RUN_TEST(test_trace);

    std::cout << "All tests passed!\n";
    return 0;
}
