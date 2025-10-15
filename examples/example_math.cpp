// examples/example_math.cpp - NumBits v0.4 Math Functions Demo
#include "numbits/numbits.hpp"
#include <iostream>
#include <iomanip>

using namespace numbits;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n==========================================\n";
    std::cout << "        NumBits v0.4 - Math Demo         \n";
    std::cout << "==========================================\n\n";

    ndarray<double> A({2, 3});
    A(0, 0) = 0.5;  A(0, 1) = 1.0;  A(0, 2) = 2.0;
    A(1, 0) = 3.0;  A(1, 1) = 4.0;  A(1, 2) = 5.0;

    std::cout << "Matrix A:\n" << A << "\n";

    // Exponential
    auto B = exp(A);
    std::cout << "exp(A):\n" << B << "\n";

    // Square root
    auto C = sqrt(A);
    std::cout << "sqrt(A):\n" << C << "\n";

    // Natural logarithm
    auto D = log(A);
    std::cout << "log(A):\n" << D << "\n";

    // Power (A^2)
    auto E = pow(A, 2.0);
    std::cout << "A^2:\n" << E << "\n";

    // Sine
    auto F = sin(A);
    std::cout << "sin(A):\n" << F << "\n";

    // Cosine
    auto G = cos(A);
    std::cout << "cos(A):\n" << G << "\n";

    // Tangent
    auto H = tan(A);
    std::cout << "tan(A):\n" << H << "\n";

    std::cout << "\n==========================================\n";
    std::cout << "          All math examples done!         \n";
    std::cout << "==========================================\n\n";

    return 0;
}
