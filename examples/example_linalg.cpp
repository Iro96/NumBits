// examples/example_linalg.cpp (v0.3 Linear Algebra Demo)
#include "numbits/numbits.hpp"
#include <iostream>
#include <iomanip>

using namespace numbits;

/**
 * @brief Demonstrates NumBits v0.3 linear algebra operations.
 * 
 * Shows usage of:
 * - matmul: Matrix multiplication
 * - trace: Sum of diagonal elements
 * - norm: Various matrix norms (Frobenius, L1, L2, infinity)
 * - det: Determinant computation
 * - inv: Matrix inversion
 * - eig: Eigenvalues and eigenvectors
 * - svd: Singular Value Decomposition
 * 
 * @return int Exit code (0 on success).
 */
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n==========================================\n";
    std::cout << "   NumBits v0.3 - Linear Algebra Demo    \n";
    std::cout << "==========================================\n\n";

    // ============================================================
    // 1. Matrix Multiplication (matmul)
    // ============================================================
    std::cout << "1. Matrix Multiplication (matmul)\n";
    std::cout << "   --------------------------------\n";
    
    ndarray<double> A({2, 3});
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    
    ndarray<double> B({3, 2});
    B(0, 0) = 7; B(0, 1) = 8;
    B(1, 0) = 9; B(1, 1) = 10;
    B(2, 0) = 11; B(2, 1) = 12;
    
    std::cout << "   A (2×3):\n" << A;
    std::cout << "   B (3×2):\n" << B;
    
    auto C = matmul(A, B);
    std::cout << "   C = A × B (2×2):\n" << C << "\n";

    // ============================================================
    // 2. Trace
    // ============================================================
    std::cout << "2. Trace (sum of diagonal elements)\n";
    std::cout << "   ---------------------------------\n";
    
    ndarray<double> M({3, 3});
    M(0, 0) = 1; M(0, 1) = 2; M(0, 2) = 3;
    M(1, 0) = 4; M(1, 1) = 5; M(1, 2) = 6;
    M(2, 0) = 7; M(2, 1) = 8; M(2, 2) = 9;
    
    std::cout << "   Matrix M:\n" << M;
    std::cout << "   trace(M) = " << trace(M) << "\n\n";

    // ============================================================
    // 3. Matrix Norms
    // ============================================================
    std::cout << "3. Matrix Norms\n";
    std::cout << "   ------------\n";
    
    ndarray<double> N({2, 2});
    N(0, 0) = 1; N(0, 1) = 2;
    N(1, 0) = 3; N(1, 1) = 4;
    
    std::cout << "   Matrix N:\n" << N;
    std::cout << "   Frobenius norm: " << norm(N, "fro") << "\n";
    std::cout << "   Infinity norm:  " << norm(N, "inf") << "\n";
    std::cout << "   1-norm:         " << norm(N, "1") << "\n\n";

    // ============================================================
    // 4. Determinant
    // ============================================================
    std::cout << "4. Determinant\n";
    std::cout << "   -----------\n";
    
    ndarray<double> D({3, 3});
    D(0, 0) = 1; D(0, 1) = 0; D(0, 2) = 2;
    D(1, 0) = -1; D(1, 1) = 3; D(1, 2) = 1;
    D(2, 0) = 2; D(2, 1) = 4; D(2, 2) = -2;
    
    std::cout << "   Matrix D:\n" << D;
    std::cout << "   det(D) = " << det(D) << "\n\n";

    // ============================================================
    // 5. Matrix Inverse
    // ============================================================
    std::cout << "5. Matrix Inverse\n";
    std::cout << "   --------------\n";
    
    ndarray<double> E({2, 2});
    E(0, 0) = 4; E(0, 1) = 7;
    E(1, 0) = 2; E(1, 1) = 6;
    
    std::cout << "   Matrix E:\n" << E;
    
    auto E_inv = inv(E);
    std::cout << "   E^(-1):\n" << E_inv;
    
    // Verify: E × E^(-1) = I
    auto I = matmul(E, E_inv);
    std::cout << "   Verification E x E^(-1) = I:\n" << I << "\n";

    // ============================================================
    // 6. Eigenvalues and Eigenvectors
    // ============================================================
    std::cout << "6. Eigenvalues and Eigenvectors\n";
    std::cout << "   ----------------------------\n";
    
    ndarray<double> S({2, 2});
    S(0, 0) = 4; S(0, 1) = 1;
    S(1, 0) = 1; S(1, 1) = 3;
    
    std::cout << "   Symmetric Matrix S:\n" << S;
    
    auto [eigenvals, eigenvecs] = eig(S);
    
    std::cout << "   Eigenvalues (as column vector):\n";
    for (size_t i = 0; i < eigenvals.shape()[0]; ++i) {
        std::cout << "   lambda" << (i+1) << " = " << eigenvals(i, 0) << "\n";
    }
    
    std::cout << "\n   Eigenvectors (as columns):\n" << eigenvecs << "\n";

    // ============================================================
    // 7. Singular Value Decomposition (SVD)
    // ============================================================
    std::cout << "7. Singular Value Decomposition\n";
    std::cout << "   ----------------------------\n";
    
    ndarray<double> X({2, 2});
    X(0, 0) = 3; X(0, 1) = 2;
    X(1, 0) = 2; X(1, 1) = 3;
    
    std::cout << "   Matrix X:\n" << X;
    
    auto [U, Sigma, Vt] = svd(X);
    
    std::cout << "   U (left singular vectors):\n" << U;
    std::cout << "   Sigma (singular values):\n" << Sigma;
    std::cout << "   V^T (right singular vectors transposed):\n" << Vt;
    
    // Reconstruct X from SVD
    auto temp = matmul(U, Sigma);
    auto X_reconstructed = matmul(temp, Vt);
    std::cout << "   Reconstruction X ~= U x Sigma x V^T:\n" << X_reconstructed << "\n";

    // ============================================================
    // Summary
    // ============================================================
    std::cout << "==========================================\n";
    std::cout << "      All examples completed!            \n";
    std::cout << "==========================================\n\n";

    return 0;
}

