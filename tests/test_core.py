#include "numbits/core/ndarray.hpp"
#include <cassert>
#include <iostream>

using namespace numbits;

int main() {
    ndarray<int> x({2, 2}, 1);
    assert(x.shape()[0] == 2);
    assert(x.shape()[1] == 2);
    std::cout << "NumBits test passed\n";
    return 0;
}
