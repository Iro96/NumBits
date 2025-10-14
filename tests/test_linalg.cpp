// tests/test_linalg.cpp - Linear Algebra Tests (v0.3)
#include "numbits/numbits.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace numbits;

// Helper function to check if two values are close
template<typename T>
bool is_close(T a, T b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// Test matmul (matrix multiplication)
void test_matmul() {
    std::cout << "Testing matmul()...\n";
    
    ndarray<double> A({2, 3});
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    
    ndarray<double> B({3, 2});
    B(0, 0) = 7; B(0, 1) = 8;
    B(1, 0) = 9; B(1, 1) = 10;
    B(2, 0) = 11; B(2, 1) = 12;
    
    auto C = matmul(A, B);
    
    assert(C.shape()[0] == 2 && C.shape()[1] == 2);
    assert(is_close(C(0, 0), 58.0));  // 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    assert(is_close(C(0, 1), 64.0));  // 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert(is_close(C(1, 0), 139.0)); // 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    assert(is_close(C(1, 1), 154.0)); // 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    
    std::cout << "matmul passed\n";
}

// Test trace
void test_trace() {
    std::cout << "Testing trace()...\n";
    
    ndarray<double> A({3, 3});
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;
    
    double tr = trace(A);
    assert(is_close(tr, 15.0)); // 1 + 5 + 9 = 15
    
    std::cout << "trace passed\n";
}

// Test norm
void test_norm() {
    std::cout << "Testing norm()...\n";
    
    ndarray<double> A({2, 2});
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    
    // Frobenius norm: sqrt(1 + 4 + 9 + 16) = sqrt(30)
    double norm_fro = norm(A, "fro");
    assert(is_close(norm_fro, std::sqrt(30.0)));
    
    // Infinity norm: max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
    double norm_inf = norm(A, "inf");
    assert(is_close(norm_inf, 7.0));
    
    // 1-norm: max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
    double norm_1 = norm(A, "1");
    assert(is_close(norm_1, 6.0));
    
    std::cout << "norm passed\n";
}

// Test determinant
void test_det() {
    std::cout << "Testing det()...\n";
    
    // 2x2 matrix
    ndarray<double> A({2, 2});
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    
    double det_A = det(A);
    assert(is_close(det_A, -2.0)); // 1*4 - 2*3 = 4 - 6 = -2
    
    // 3x3 matrix
    ndarray<double> B({3, 3});
    B(0, 0) = 1; B(0, 1) = 0; B(0, 2) = 2;
    B(1, 0) = -1; B(1, 1) = 3; B(1, 2) = 1;
    B(2, 0) = 2; B(2, 1) = 4; B(2, 2) = -2;
    
    double det_B = det(B);
    // det = 1*(3*(-2) - 1*4) - 0 + 2*(-1*4 - 3*2)
    //     = 1*(-6 - 4) + 2*(-4 - 6)
    //     = -10 + 2*(-10) = -10 - 20 = -30
    assert(is_close(det_B, -30.0, 1e-5));
    
    // Identity matrix
    ndarray<double> I({3, 3}, 0.0);
    I(0, 0) = I(1, 1) = I(2, 2) = 1.0;
    double det_I = det(I);
    assert(is_close(det_I, 1.0));
    
    std::cout << "det passed\n";
}

// Test matrix inverse
void test_inv() {
    std::cout << "Testing inv()...\n";
    
    // 2x2 matrix
    ndarray<double> A({2, 2});
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    
    auto A_inv = inv(A);
    
    // Expected inverse: (1/-2) * [[4, -2], [-3, 1]] = [[-2, 1], [1.5, -0.5]]
    assert(is_close(A_inv(0, 0), -2.0));
    assert(is_close(A_inv(0, 1), 1.0));
    assert(is_close(A_inv(1, 0), 1.5));
    assert(is_close(A_inv(1, 1), -0.5));
    
    // Verify A * A_inv = I
    auto I = matmul(A, A_inv);
    assert(is_close(I(0, 0), 1.0, 1e-5));
    assert(is_close(I(0, 1), 0.0, 1e-5));
    assert(is_close(I(1, 0), 0.0, 1e-5));
    assert(is_close(I(1, 1), 1.0, 1e-5));
    
    std::cout << "inv passed\n";
}

// Test eigenvalues and eigenvectors
void test_eig() {
    std::cout << "Testing eig()...\n";
    
    // Symmetric 2x2 matrix (easier to verify)
    ndarray<double> A({2, 2});
    A(0, 0) = 4; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 3;
    
    auto [eigenvals, eigenvecs] = eig(A, 1000, 1e-8);
    
    // Verify A * v = lambda * v for each eigenvector
    for (size_t i = 0; i < 2; ++i) {
        double lambda = eigenvals(i, 0);
        
        // Extract eigenvector
        ndarray<double> v({2, 1});
        v(0, 0) = eigenvecs(0, i);
        v(1, 0) = eigenvecs(1, i);
        
        // Compute A * v
        ndarray<double> Av({2, 1}, 0.0);
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                Av(j, 0) += A(j, k) * v(k, 0);
            }
        }
        
        // Compute lambda * v
        ndarray<double> lambda_v({2, 1});
        lambda_v(0, 0) = lambda * v(0, 0);
        lambda_v(1, 0) = lambda * v(1, 0);
        
        // Check if Av ≈ lambda * v
        assert(is_close(Av(0, 0), lambda_v(0, 0), 1e-4));
        assert(is_close(Av(1, 0), lambda_v(1, 0), 1e-4));
    }
    
    std::cout << "eig passed\n";
}

// Test SVD
void test_svd() {
    std::cout << "Testing svd()...\n";
    
    // Simple 2x2 matrix
    ndarray<double> A({2, 2});
    A(0, 0) = 3; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 2;
    
    auto [U, S, Vt] = svd(A, 1000, 1e-8);
    
    // For diagonal matrix, singular values should be |diagonal elements|
    // Just verify dimensions are correct
    assert(U.shape()[0] == 2 && U.shape()[1] == 2);
    assert(S.shape()[0] == 2 && S.shape()[1] == 2);
    assert(Vt.shape()[0] == 2 && Vt.shape()[1] == 2);
    
    // Verify that U * S * Vt ≈ A (basic reconstruction)
    auto temp = matmul(U, S);
    auto A_reconstructed = matmul(temp, Vt);
    
    // Check reconstruction (with relaxed tolerance due to iterative algorithm)
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            assert(is_close(A_reconstructed(i, j), A(i, j), 1e-2));
        }
    }
    
    std::cout << "svd passed\n";
}

int main() {
    std::cout << "\n=== NumBits v0.3 Linear Algebra Tests ===\n\n";
    
    try {
        test_matmul();
        test_trace();
        test_norm();
        test_det();
        test_inv();
        test_eig();
        test_svd();
        
        std::cout << "\nAll linear algebra tests passed!\n\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nTest failed with error: " << e.what() << "\n\n";
        return 1;
    }
}

