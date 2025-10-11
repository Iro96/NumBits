#include "numbits/core/ndarray.hpp"
#include <cassert>
#include <iostream>

using namespace numbits;

int main() {
    // Basic shape + element check
    ndarray<int> a({2, 3}, 5);

    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 3);
    assert(a(1, 2) == 5);

    a(1, 2) = 42;
    assert(a(1, 2) == 42);

    std::cout << "NumBits basic test passed.\n";
    return 0;
}
