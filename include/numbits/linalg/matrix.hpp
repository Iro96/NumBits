#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>

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

} // namespace numbits
