#include <iostream>
#include <cassert>
#include "numbits/numbits.hpp"

using namespace numbits;

#define TEST_CASE(name) void name()
#define RUN_TEST(name)  \
    std::cout << "Running " #name "... "; \
    name(); \
    std::cout << "OK\n";

TEST_CASE(test_addition) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    auto c = a + b;
    assert(c[0] == 6.0f);
    assert(c[1] == 8.0f);
    assert(c[2] == 10.0f);
    assert(c[3] == 12.0f);
}

TEST_CASE(test_scalar_addition) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = a + 5.0f;
    assert(b[0] == 6.0f);
    assert(b[1] == 7.0f);
    assert(b[2] == 8.0f);
    assert(b[3] == 9.0f);
}

TEST_CASE(test_multiplication) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> b({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    auto c = a * b;
    assert(c[0] == 2.0f);
    assert(c[1] == 4.0f);
    assert(c[2] == 6.0f);
    assert(c[3] == 8.0f);
}

TEST_CASE(test_sum_reduction) {
    ndarray<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    assert(sum(a) == 10.0f);
}

TEST_CASE(test_mean_reduction) {
    ndarray<float> a({2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
    assert(mean(a) == 5.0f);
}

int main() {
    RUN_TEST(test_addition);
    RUN_TEST(test_scalar_addition);
    RUN_TEST(test_multiplication);
    RUN_TEST(test_sum_reduction);
    RUN_TEST(test_mean_reduction);

    std::cout << "All tests passed!\n";
    return 0;
}
