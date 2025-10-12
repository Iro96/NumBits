// examples/example_basic.cpp (v0.2 test)
#include "numbits/numbits.hpp"
#include <iostream>

using namespace numbits;

/**
 * @brief Runs a self-contained example that creates two random 2x3 tensors, computes their element-wise sum, exercises additional tensor operations (reshape, transpose, expand_dims, squeeze, broadcast_to, slice), and prints each tensor to standard output.
 *
 * @return int 0 on success.
 */
int main() {
    auto A = rand<double>({2, 3});
    auto B = rand<double>({2, 3});
    auto C = add(A, B);

    std::cout << "A:\n" << A;
    std::cout << "B:\n" << B;
    std::cout << "A + B:\n" << C;

    // --- v0.2 tests ---
    auto D = reshape(C, {3, 2});
    std::cout << "Reshape (3x2):\n" << D;

    auto E = transpose(D);
    std::cout << "Transpose:\n" << E;

    auto F = expand_dims(A, 0);
    std::cout << "Expand dims axis 0:\n" << F;

    auto G = squeeze(F, 0);
    std::cout << "Squeeze axis 0:\n" << G;

    auto H = broadcast_to(A, {2, 3});
    std::cout << "Broadcast to (2x3):\n" << H;

    auto I = slice(A, 0, 2, 1, 3);
    std::cout << "Slice (rows 0-2, cols 1-3):\n" << I;
}