#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>
#include "../core/ndarray.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numbits {

/**
 * @brief Matrix multiplication (dot product) for 2D ndarrays.
 * 
 * Computes: C = A × B
 * where A is (n × p) and B is (p × m)
 * 
 * @throws std::invalid_argument if input arrays are not 2D or have mismatched inner dimensions.
 */
template <typename T>
ndarray<T> dot(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& sA = A.shape();
    const auto& sB = B.shape();

    // Validate ranks (only 2D arrays supported)
    if (sA.size() != 2 || sB.size() != 2)
        throw std::invalid_argument("dot: expected two 2D ndarrays");

    const size_t n = sA[0];
    const size_t p = sA[1];
    const size_t q = sB[0];
    const size_t m = sB[1];

    // Validate inner dimension match
    if (p != q)
        throw std::invalid_argument("dot: inner dimensions must match");

    // Initialize output matrix with zeros
    ndarray<T> C({n, m}, T{});

    // Standard triple-nested loop (can later optimize with OpenMP or BLAS)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            for (size_t k = 0; k < p; ++k)
                C(i, j) += A(i, k) * B(k, j);

    return C;
}

/**
 * @brief Matrix multiplication (alias for dot).
 * 
 * Computes: C = A × B
 * 
 * @throws std::invalid_argument if input arrays are not 2D or have mismatched inner dimensions.
 */
template <typename T>
ndarray<T> matmul(const ndarray<T>& A, const ndarray<T>& B) {
    return dot(A, B);
}

/**
 * @brief Compute the trace of a 2D matrix (sum of diagonal elements).
 * 
 * @param A A 2D square matrix
 * @return T Sum of diagonal elements
 * @throws std::invalid_argument if A is not 2D or not square
 */
template <typename T>
T trace(const ndarray<T>& A) {
    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("trace: expected 2D ndarray");
    if (s[0] != s[1])
        throw std::invalid_argument("trace: expected square matrix");
    
    T sum = T{};
    for (size_t i = 0; i < s[0]; ++i)
        sum += A(i, i);
    
    return sum;
}

/**
 * @brief Compute matrix or vector norm.
 * 
 * Supported norms:
 * - "fro" or "f": Frobenius norm (sqrt of sum of squares of all elements)
 * - "inf": Maximum absolute row sum
 * - "1": Maximum absolute column sum
 * - "2": Spectral norm (largest singular value) - approximate for now
 * 
 * @param A Input 2D matrix
 * @param ord Norm type ("fro", "inf", "1", "2")
 * @return Computed norm value
 * @throws std::invalid_argument if A is not 2D or ord is invalid
 */
template <typename T>
double norm(const ndarray<T>& A, const std::string& ord = "fro") {
    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("norm: expected 2D ndarray");
    
    const size_t rows = s[0];
    const size_t cols = s[1];
    
    if (ord == "fro" || ord == "f") {
        // Frobenius norm: sqrt(sum of all squared elements)
        double sum_sq = 0.0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                double val = static_cast<double>(A(i, j));
                sum_sq += val * val;
            }
        return std::sqrt(sum_sq);
    }
    else if (ord == "inf") {
        // Infinity norm: max absolute row sum
        double max_sum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < cols; ++j)
                row_sum += std::abs(static_cast<double>(A(i, j)));
            max_sum = std::max(max_sum, row_sum);
        }
        return max_sum;
    }
    else if (ord == "1") {
        // 1-norm: max absolute column sum
        double max_sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            double col_sum = 0.0;
            for (size_t i = 0; i < rows; ++i)
                col_sum += std::abs(static_cast<double>(A(i, j)));
            max_sum = std::max(max_sum, col_sum);
        }
        return max_sum;
    }
    else if (ord == "2") {
        // 2-norm (spectral norm): For now, use Frobenius as approximation
        // Full implementation would require SVD
        return norm(A, "fro");
    }
    else {
        throw std::invalid_argument("norm: unsupported norm type '" + ord + "'");
    }
}

/**
 * @brief Compute determinant of a square matrix using LU decomposition.
 * 
 * @param A Square matrix
 * @return Determinant value
 * @throws std::invalid_argument if A is not 2D or not square
 */
template <typename T>
T det(const ndarray<T>& A) {
    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("det: expected 2D ndarray");
    if (s[0] != s[1])
        throw std::invalid_argument("det: expected square matrix");
    
    const size_t n = s[0];
    
    // Special cases
    if (n == 1) return A(0, 0);
    if (n == 2) return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    
    // Create a copy for LU decomposition
    ndarray<T> LU({n, n});
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            LU(i, j) = A(i, j);
    
    // LU decomposition with partial pivoting
    T det_value = T{1};
    int num_swaps = 0;
    
    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t pivot = k;
        T max_val = std::abs(LU(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            T abs_val = std::abs(LU(i, k));
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot = i;
            }
        }
        
        // Swap rows if needed
        if (pivot != k) {
            for (size_t j = 0; j < n; ++j)
                std::swap(LU(k, j), LU(pivot, j));
            num_swaps++;
        }
        
        // Check for singular matrix
        if (std::abs(LU(k, k)) < std::numeric_limits<T>::epsilon() * 10)
            return T{0};
        
        // Eliminate column
        for (size_t i = k + 1; i < n; ++i) {
            T factor = LU(i, k) / LU(k, k);
            for (size_t j = k + 1; j < n; ++j)
                LU(i, j) -= factor * LU(k, j);
        }
        
        det_value *= LU(k, k);
    }
    
    // Account for row swaps
    return (num_swaps % 2 == 0) ? det_value : -det_value;
}

/**
 * @brief Compute matrix inverse using Gauss-Jordan elimination.
 * 
 * @param A Square matrix
 * @return Inverse of A
 * @throws std::invalid_argument if A is not 2D, not square, or is singular
 */
template <typename T>
ndarray<T> inv(const ndarray<T>& A) {
    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("inv: expected 2D ndarray");
    if (s[0] != s[1])
        throw std::invalid_argument("inv: expected square matrix");
    
    const size_t n = s[0];
    
    // Create augmented matrix [A | I]
    ndarray<T> aug({n, 2 * n}, T{});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            aug(i, j) = A(i, j);
        aug(i, n + i) = T{1};  // Identity on right side
    }
    
    // Gauss-Jordan elimination
    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t pivot = k;
        T max_val = std::abs(aug(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            T abs_val = std::abs(aug(i, k));
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot = i;
            }
        }
        
        // Swap rows
        if (pivot != k) {
            for (size_t j = 0; j < 2 * n; ++j)
                std::swap(aug(k, j), aug(pivot, j));
        }
        
        // Check for singularity
        if (std::abs(aug(k, k)) < std::numeric_limits<T>::epsilon() * 10)
            throw std::runtime_error("inv: matrix is singular");
        
        // Scale pivot row
        T pivot_val = aug(k, k);
        for (size_t j = 0; j < 2 * n; ++j)
            aug(k, j) /= pivot_val;
        
        // Eliminate column
        for (size_t i = 0; i < n; ++i) {
            if (i != k) {
                T factor = aug(i, k);
                for (size_t j = 0; j < 2 * n; ++j)
                    aug(i, j) -= factor * aug(k, j);
            }
        }
    }
    
    // Extract inverse from right half
    ndarray<T> inv_A({n, n});
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            inv_A(i, j) = aug(i, n + j);
    
    return inv_A;
}

/**
 * @brief Compute eigenvalues and eigenvectors using QR algorithm.
 * 
 * Returns a pair: (eigenvalues, eigenvectors)
 * - eigenvalues: 1D array of eigenvalues (stored as 2D nx1 for consistency)
 * - eigenvectors: 2D array where each column is an eigenvector
 * 
 * Note: This is a simplified implementation. For production use, consider using
 * specialized libraries like Eigen or LAPACK.
 * 
 * @param A Square matrix
 * @param max_iter Maximum number of iterations (default: 1000)
 * @param tol Convergence tolerance (default: 1e-10)
 * @return std::pair<ndarray<T>, ndarray<T>> (eigenvalues, eigenvectors)
 * @throws std::invalid_argument if A is not 2D or not square
 */
template <typename T>
std::pair<ndarray<T>, ndarray<T>> eig(const ndarray<T>& A, size_t max_iter = 1000, double tol = 1e-10) {
    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("eig: expected 2D ndarray");
    if (s[0] != s[1])
        throw std::invalid_argument("eig: expected square matrix");
    
    const size_t n = s[0];
    
    // Initialize with copy of A
    ndarray<T> Ak({n, n});
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Ak(i, j) = A(i, j);
    
    // Initialize eigenvector matrix as identity
    ndarray<T> V({n, n}, T{});
    for (size_t i = 0; i < n; ++i)
        V(i, i) = T{1};
    
    // QR algorithm iterations
    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Simple QR decomposition using Gram-Schmidt
        ndarray<T> Q({n, n}, T{});
        ndarray<T> R({n, n}, T{});
        
        // Gram-Schmidt orthogonalization
        for (size_t j = 0; j < n; ++j) {
            // Copy column j from Ak
            std::vector<T> v(n);
            for (size_t i = 0; i < n; ++i)
                v[i] = Ak(i, j);
            
            // Orthogonalize against previous columns
            for (size_t k = 0; k < j; ++k) {
                T dot_prod = T{};
                for (size_t i = 0; i < n; ++i)
                    dot_prod += Q(i, k) * v[i];
                R(k, j) = dot_prod;
                for (size_t i = 0; i < n; ++i)
                    v[i] -= dot_prod * Q(i, k);
            }
            
            // Normalize
            T norm_v = T{};
            for (size_t i = 0; i < n; ++i)
                norm_v += v[i] * v[i];
            norm_v = std::sqrt(norm_v);
            
            if (norm_v < tol)
                norm_v = T{1};  // Avoid division by zero
            
            R(j, j) = norm_v;
            for (size_t i = 0; i < n; ++i)
                Q(i, j) = v[i] / norm_v;
        }
        
        // Ak+1 = R * Q
        ndarray<T> Ak_new({n, n}, T{});
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    Ak_new(i, j) += R(i, k) * Q(k, j);
        
        // Update eigenvectors: V = V * Q
        ndarray<T> V_new({n, n}, T{});
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    V_new(i, j) += V(i, k) * Q(k, j);
        
        // Check convergence (off-diagonal elements should be small)
        double off_diag_norm = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (i != j)
                    off_diag_norm += std::abs(static_cast<double>(Ak_new(i, j)));
        
        Ak = Ak_new;
        V = V_new;
        
        if (off_diag_norm < tol)
            break;
    }
    
    // Extract eigenvalues from diagonal
    ndarray<T> eigenvalues({n, 1});
    for (size_t i = 0; i < n; ++i)
        eigenvalues(i, 0) = Ak(i, i);
    
    return {eigenvalues, V};
}

/**
 * @brief Compute Singular Value Decomposition (SVD): A = U * S * V^T
 * 
 * Returns a tuple: (U, S, Vt)
 * - U: Left singular vectors (m × m orthogonal matrix)
 * - S: Singular values as diagonal matrix (m × n)
 * - Vt: Right singular vectors transposed (n × n orthogonal matrix)
 * 
 * This is a simplified implementation using the relationship:
 * - A^T * A has eigenvalues = singular values squared, eigenvectors = V
 * - A * V = U * S
 * 
 * @param A Input matrix (m × n)
 * @param max_iter Maximum iterations for eigenvalue computation
 * @param tol Convergence tolerance
 * @return std::tuple<ndarray<T>, ndarray<T>, ndarray<T>> (U, S, Vt)
 * @throws std::invalid_argument if A is not 2D
 */
template <typename T>
std::tuple<ndarray<T>, ndarray<T>, ndarray<T>> svd(const ndarray<T>& A, size_t max_iter = 1000, double tol = 1e-10) {
    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("svd: expected 2D ndarray");
    
    const size_t m = s[0];
    const size_t n = s[1];
    
    // Compute A^T * A
    ndarray<T> AtA({n, n}, T{});
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < m; ++k)
                AtA(i, j) += A(k, i) * A(k, j);
    
    // Compute eigenvalues and eigenvectors of A^T * A
    auto [eigenvals_sq, V] = eig(AtA, max_iter, tol);
    
    // Sort eigenvalues and eigenvectors in descending order
    std::vector<std::pair<T, size_t>> sorted_eigs;
    for (size_t i = 0; i < n; ++i)
        sorted_eigs.push_back({eigenvals_sq(i, 0), i});
    std::sort(sorted_eigs.begin(), sorted_eigs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Reorder V and compute singular values
    ndarray<T> V_sorted({n, n});
    ndarray<T> S({m, n}, T{});
    const size_t r = std::min(m, n);
    for (size_t i = 0; i < r; ++i) {
         size_t idx = sorted_eigs[i].second;
         T sing_val = std::sqrt(std::max(T{0}, sorted_eigs[i].first));
         S(i, i) = sing_val;
         for (size_t j = 0; j < n; ++j)
             V_sorted(j, i) = V(j, idx);
     }
    
    // Compute U = A * V * S^{-1}
    ndarray<T> U({m, m}, T{});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < std::min(m, n); ++j) {
            T sum = T{};
            for (size_t k = 0; k < n; ++k) {
                sum += A(i, k) * V_sorted(k, j);
            }
            if (S(j, j) > tol)
                U(i, j) = sum / S(j, j);
            else
                U(i, j) = T{};
        }
    }
    
    // Complete U to orthonormal basis if needed (simplified - just fill with identity)
    for (size_t i = std::min(m, n); i < m; ++i)
        U(i, i) = T{1};
    
    // Transpose V for Vt
    ndarray<T> Vt({n, n});
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Vt(i, j) = V_sorted(j, i);
    
    return {U, S, Vt};
}

} // namespace numbits
