#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>

namespace numbits {

/**
 * @brief Elementwise addition: C = A + B
 */
template <typename T>
ndarray<T> add(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();

    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("add: expected 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("add: shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            C(i, j) = A(i, j) + B(i, j);
    return C;
}

/**
 * @brief Elementwise subtraction: C = A - B
 */
template <typename T>
ndarray<T> sub(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();

    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("sub: expected 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("sub: shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            C(i, j) = A(i, j) - B(i, j);
    return C;
}

/**
 * @brief Elementwise multiplication: C = A ⊙ B
 */
template <typename T>
ndarray<T> mul(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();

    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("mul: expected 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("mul: shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            C(i, j) = A(i, j) * B(i, j);
    return C;
}

/**
 * @brief Elementwise division with zero guard: C = A / B
 * @throws std::domain_error if B(i, j) == 0
 */
template <typename T>
ndarray<T> div(const ndarray<T>& A, const ndarray<T>& B) {
    const auto& as = A.shape();
    const auto& bs = B.shape();

    if (as.size() != 2 || bs.size() != 2)
        throw std::invalid_argument("div: expected 2D ndarrays");
    if (as != bs)
        throw std::invalid_argument("div: shape mismatch");

    const size_t rows = as[0], cols = as[1];
    ndarray<T> C({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            if (B(i, j) == T(0))
                throw std::domain_error("div: division by zero");
            C(i, j) = A(i, j) / B(i, j);
        }
    return C;
}

} // namespace numbits
