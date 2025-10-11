#pragma once
#include "../core/ndarray.hpp"
#include <cmath>
#include <type_traits>
#include <stdexcept>

namespace numbits {

/**
 * @brief Elementwise exponential: B = exp(A)
 */
template <typename T>
ndarray<T> exp(const ndarray<T>& A) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::exp requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("exp: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            B(i, j) = std::exp(A(i, j));

    return B;
}

/**
 * @brief Elementwise square root: B = sqrt(A)
 */
template <typename T>
ndarray<T> sqrt(const ndarray<T>& A) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::sqrt requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("sqrt: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            if (A(i, j) < T(0))
                throw std::domain_error("sqrt: negative input value");
            B(i, j) = std::sqrt(A(i, j));
        }

    return B;
}

} // namespace numbits
