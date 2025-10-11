#include "numbits/numbits.hpp"
#include <cassert>
#include <iostream>

using namespace numbits;

int main() {
    ndarray<int> A({2, 3}, 2);
    ndarray<int> B({2, 3}, 3);
    auto C = add(A, B);

    assert(C(0, 0) == 5);
    assert(sum(C) == 30);
    std::cout << "test_core passed" << std::endl;
}
