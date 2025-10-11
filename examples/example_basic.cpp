#include "numbits/numbits.hpp"
#include <iostream>

using namespace numbits;

int main() {
    auto A = rand<double>({2, 3});
    auto B = rand<double>({2, 3});
    auto C = add(A, B);

    std::cout << "A:\n" << A;
    std::cout << "B:\n" << B;
    std::cout << "A + B:\n" << C;
}
