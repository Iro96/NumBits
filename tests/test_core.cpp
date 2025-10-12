#include "numbits/numbits.hpp"
#include <cassert>
#include <iostream>

using namespace numbits;

int main() {
    // --- Arithmetic test ---
    ndarray<int> A({2, 3}, 2);
    ndarray<int> B({2, 3}, 3);
    auto C = add(A, B);

    assert(C(0, 0) == 5);
    assert(sum(C) == 30);

    // --- Reshape test ---
    auto D = reshape(C, {3, 2});
    assert(D.shape() == std::vector<size_t>({3, 2}));
    assert(D(0, 0) == 5);

    // --- Transpose test ---
    auto E = transpose(D);
    assert(E.shape() == std::vector<size_t>({2, 3}));
    assert(E(0, 0) == 5);

    // --- Expand dims test ---
    auto F = expand_dims(E, 0);
    assert(F.shape() == std::vector<size_t>({1, 2, 3}));

    // --- Broadcast test ---
    auto G = broadcast_to(A, {2, 3});
    assert(G.shape() == std::vector<size_t>({2, 3}));

    std::cout << "test_core v0.2 passed" << std::endl;
}
