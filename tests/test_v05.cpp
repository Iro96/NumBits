#include "numbits/numbits.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <cmath>

using namespace numbits;

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

    auto assert_almost_equal = [](double a, double b, double tol=1e-5){
        assert(std::abs(a-b) < tol);
    };

    // 1D
    ndarray<double> X1({5}, {1,2,3,4,5});
    assert_almost_equal(mean(X1), 3.0);
    assert_almost_equal(variance(X1), 2.0);

    // 4D
    std::vector<size_t> dims4{2,2,2,3};
    std::vector<double> data4(total_size(dims4));
    for(size_t i=0;i<data4.size();i++) data4[i]=i+1;
    auto X4 = make_ndarray(dims4, data4);
    assert_almost_equal(mean(X4), sum(data4)/double(total_size(dims4)));

    // 8D
    std::vector<size_t> dims8{2,2,2,2,1,1,1,3};
    std::vector<double> data8(total_size(dims8));
    for(size_t i=0;i<data8.size();i++) data8[i]=i+1;
    auto X8 = make_ndarray(dims8, data8);
    assert_almost_equal(mean(X8), sum(data8)/double(total_size(dims8)));

    // 10D
    std::vector<size_t> dims10{2,1,1,1,1,1,1,1,2,3};
    std::vector<double> data10(total_size(dims10));
    for(size_t i=0;i<data10.size();i++) data10[i]=i+1;
    auto X10 = make_ndarray(dims10, data10);
    assert_almost_equal(mean(X10), sum(data10)/double(total_size(dims10)));

    // 14D
    std::vector<size_t> dims14{1,1,1,1,1,1,1,1,1,1,1,1,2,3};
    std::vector<double> data14(total_size(dims14));
    for(size_t i=0;i<data14.size();i++) data14[i]=i+1;
    auto X14 = make_ndarray(dims14, data14);
    assert_almost_equal(mean(X14), sum(data14)/double(total_size(dims14)));

    // 100D
    std::vector<size_t> dims100(100,1);
    std::vector<double> data100(total_size(dims100), 42.0);
    auto X100 = make_ndarray(dims100, data100);
    assert_almost_equal(mean(X100), 42.0);

    std::cout << "N-D statistical tests passed.\n";
    std::cout << "All tests for NumBits v0.5 passed successfully.\n";
    return 0;
}
