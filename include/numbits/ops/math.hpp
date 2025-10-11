#pragma once
#include "../core/ndarray.hpp"
#include <cmath>

namespace numbits {

template <typename T>
ndarray<T> exp(const ndarray<T>& A) {
    ndarray<T> B({A.shape()[0], A.shape()[1]});
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < A.shape()[1]; ++j)
            B(i, j) = std::exp(A(i, j));
    return B;
}

template <typename T>
ndarray<T> sqrt(const ndarray<T>& A) {
    ndarray<T> B({A.shape()[0], A.shape()[1]});
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < A.shape()[1]; ++j)
            B(i, j) = std::sqrt(A(i, j));
    return B;
}

} // namespace nb
