#include "numbits/core/ndarray.hpp"
#include <cassert>
#include <iostream>

using namespace numbits;

int main() {
    try {
        // Create 2x3 ndarray of ints, initialized to 5
        ndarray<int> a({2, 3}, 5);

        assert(a.shape()[0] == 2);
        assert(a.shape()[1] == 3);
        assert(a(0, 0) == 5);
        assert(a(1, 2) == 5);

        // Modify a value and check
        a(1, 1) = 99;
        assert(a(1, 1) == 99);

        std::cout << "NumBits test_core passed successfully.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}
