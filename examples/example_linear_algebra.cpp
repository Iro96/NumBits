#include <iostream>
#include "numbits/ndarray.hpp"
#include "numbits/linear_algebra.hpp"

using namespace numbits;

int main() {
    std::cout << "=== NumBits Linear Algebra Example ===\n\n";
    
    // Create matrices
    ndarray<float> A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ndarray<float> B({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    std::cout << "Matrix A (2x3):\n";
    A.print();
    std::cout << "\n\n";
    
    std::cout << "Matrix B (3x2):\n";
    B.print();
    std::cout << "\n\n";
    
    // Matrix multiplication
    auto C = matmul(A, B);
    std::cout << "A @ B (matrix multiplication):\n";
    C.print();
    std::cout << "\n\n";
    
    // Transpose
    auto At = transpose(A);
    std::cout << "A transpose:\n";
    At.print();
    std::cout << "\n\n";
    
    // Determinant
    ndarray<float> square({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "Square matrix:\n";
    square.print();
    std::cout << "\n";
    std::cout << "Determinant: " << determinant(square) << "\n\n";
    
    // Inverse
    ndarray<float> invertible({2, 2}, {4.0f, 7.0f, 2.0f, 6.0f});
    std::cout << "Invertible matrix:\n";
    invertible.print();
    std::cout << "\n";
    
    auto inv = inverse(invertible);
    std::cout << "Inverse:\n";
    inv.print();
    std::cout << "\n\n";
    
    // Verify: A @ A^(-1) should be identity
    auto identity_check = matmul(invertible, inv);
    std::cout << "A @ A^(-1) (should be identity):\n";
    identity_check.print();
    std::cout << "\n\n";
    
    // Trace
    ndarray<float> matrix({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    std::cout << "Matrix:\n";
    matrix.print();
    std::cout << "\n";
    std::cout << "Trace: " << trace(matrix) << "\n";
    
    return 0;
}

