#pragma once
#include "../core/ndarray.hpp"

namespace numbits {

template <typename T>
ndarray<T> dot(const ndarray<T>& A, const ndarray<T>& B) {
    size_t n = A.shape()[0], m = B.shape()[1], p = A.shape()[1];
    ndarray<T> C({n, m}, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            for (size_t k = 0; k < p; ++k)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

} // namespace nb
