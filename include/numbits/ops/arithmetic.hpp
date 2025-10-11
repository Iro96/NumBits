#pragma once
#include "../core/ndarray.hpp"

namespace numbits {

template <typename T>
ndarray<T> add(const ndarray<T>& A, const ndarray<T>& B) {
    ndarray<T> C({A.shape()[0], A.shape()[1]});
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < A.shape()[1]; ++j)
            C(i, j) = A(i, j) + B(i, j);
    return C;
}

template <typename T>
ndarray<T> sub(const ndarray<T>& A, const ndarray<T>& B) {
    ndarray<T> C({A.shape()[0], A.shape()[1]});
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < A.shape()[1]; ++j)
            C(i, j) = A(i, j) - B(i, j);
    return C;
}

template <typename T>
ndarray<T> mul(const ndarray<T>& A, const ndarray<T>& B) {
    ndarray<T> C({A.shape()[0], A.shape()[1]});
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < A.shape()[1]; ++j)
            C(i, j) = A(i, j) * B(i, j);
    return C;
}

template <typename T>
ndarray<T> div(const ndarray<T>& A, const ndarray<T>& B) {
    ndarray<T> C({A.shape()[0], A.shape()[1]});
    for (size_t i = 0; i < A.shape()[0]; ++i)
        for (size_t j = 0; j < A.shape()[1]; ++j)
            C(i, j) = A(i, j) / B(i, j);
    return C;
}

} // namespace nb
