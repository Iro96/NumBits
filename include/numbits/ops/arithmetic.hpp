#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>

namespace numbits {

// Elementwise addition
template <typename T>
ndarray<T> add(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();
    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("add expects 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("add shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            C(i, j) = A(i, j) + B(i, j);
    return C;
}

// Elementwise subtraction
template <typename T>
ndarray<T> sub(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();
    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("sub expects 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("sub shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            C(i, j) = A(i, j) - B(i, j);
    return C;
}

// Elementwise multiplication
template <typename T>
ndarray<T> mul(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();
    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("mul expects 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("mul shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            C(i, j) = A(i, j) * B(i, j);
    return C;
}

// Elementwise division (with zero guard)
template <typename T>
ndarray<T> div(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();
    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("div expects 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("div shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            if (B(i, j) == T(0))
                throw std::domain_error("division by zero in ndarray::div");
            C(i, j) = A(i, j) / B(i, j);
        }
    return C;
}

} // namespace numbits
