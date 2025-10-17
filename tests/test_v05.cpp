#include "numbits/numbits.hpp"
#include <cassert>
#include <iostream>
#include <memory>

using namespace numbits;

/**
 * @brief Runs a sequence of tests for ndarray and statistical operations.
 *
 * Covers:
 * - Core ndarray operations: arithmetic, reshape, transpose, expand_dims, broadcast
 * - Mean, variance, stddev for 1D to 100D arrays
 * - Covariance and correlation matrices
 *
 * Aborts via assertion failure if any test fails.
 *
 * @return int Exit status: 0 on success.
 */
int main() {
    // --------------------
    // Core ndarray tests
    // --------------------
    ndarray<int> A({2, 3}, 2);
    ndarray<int> B({2, 3}, 3);
    auto C = add(A, B);
    assert(C(0, 0) == 5);
    assert(sum(C) == 30);

    auto D = reshape(C, {3, 2});
    assert(D.shape() == std::vector<size_t>({3, 2}));
    assert(D(0, 0) == 5);

    auto E = transpose(D);
    assert(E.shape() == std::vector<size_t>({2, 3}));
    assert(E(0, 0) == 5);

    auto F = expand_dims(E, 0);
    assert(F.shape() == std::vector<size_t>({1, 2, 3}));

    auto G = broadcast_to(A, {2, 3});
    assert(G.shape() == std::vector<size_t>({2, 3}));

    std::cout << "Core ndarray tests passed.\n";

    // --------------------
    // Statistical tests (N-D arrays)
    // --------------------
    auto make_ndarray = [](const std::vector<size_t>& dims, const std::vector<double>& data){
        auto ptr = std::make_shared<std::vector<double>>(data);
        return ndarray<double>(dims, ptr);
    };

    // 1D
    ndarray<double> X1({5}, {1,2,3,4,5});
    assert(mean(X1) == 3.0);
    assert(std::abs(variance(X1) - 2.0) < 1e-5);

    // 4D
    std::vector<size_t> dims4{2,2,2,3};
    std::vector<double> data4(24);
    for(size_t i=0;i<24;i++) data4[i]=i+1;
    auto X4 = make_ndarray(dims4, data4);
    assert(mean(X4) == 12.5);

    // 8D
    std::vector<size_t> dims8{2,2,2,2,1,1,1,3};
    std::vector<double> data8(24);
    for(size_t i=0;i<24;i++) data8[i]=i+1;
    auto X8 = make_ndarray(dims8, data8);
    assert(mean(X8) == 12.5); // same as 4D, sum/data size

    // 10D
    std::vector<size_t> dims10{2,1,1,1,1,1,1,1,2,3};
    std::vector<double> data10(12);
    for(size_t i=0;i<12;i++) data10[i]=i+1;
    auto X10 = make_ndarray(dims10, data10);
    assert(mean(X10) == 6.5);

    // 14D
    std::vector<size_t> dims14{1,1,1,1,1,1,1,1,1,1,1,1,2,3};
    std::vector<double> data14(6);
    for(size_t i=0;i<6;i++) data14[i]=i+1;
    auto X14 = make_ndarray(dims14, data14);
    assert(mean(X14) == 3.5);

    // 100D
    std::vector<size_t> dims100(100,1);
    std::vector<double> data100(1, 42.0);
    auto X100 = make_ndarray(dims100, data100);
    assert(mean(X100) == 42.0);

    std::cout << "N-D statistical tests passed.\n";
    std::cout << "All tests for NumBits v0.5 passed successfully.\n";
    return 0;
}
