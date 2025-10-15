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

/**
 * @brief Elementwise natural logarithm: B = log(A)
 */
template <typename T>
ndarray<T> log(const ndarray<T>& A) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::log requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("log: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            if (A(i, j) <= T(0))
                throw std::domain_error("log: input must be positive");
            B(i, j) = std::log(A(i, j));
        }

    return B;
}

/**
 * @brief Elementwise power: B = A^exponent
 */
template <typename T>
ndarray<T> pow(const ndarray<T>& A, T exponent) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::pow requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("pow: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            B(i, j) = std::pow(A(i, j), exponent);

    return B;
}

/**
 * @brief Elementwise sine: B = sin(A)
 */
template <typename T>
ndarray<T> sin(const ndarray<T>& A) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::sin requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("sin: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            B(i, j) = std::sin(A(i, j));

    return B;
}

/**
 * @brief Elementwise cosine: B = cos(A)
 */
template <typename T>
ndarray<T> cos(const ndarray<T>& A) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::cos requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("cos: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            B(i, j) = std::cos(A(i, j));

    return B;
}

/**
 * @brief Elementwise tangent: B = tan(A)
 */
template <typename T>
ndarray<T> tan(const ndarray<T>& A) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::tan requires floating-point T");

    const auto& s = A.shape();
    if (s.size() != 2)
        throw std::invalid_argument("tan: expected a 2D ndarray");

    const size_t rows = s[0], cols = s[1];
    ndarray<T> B({rows, cols});

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            B(i, j) = std::tan(A(i, j));

    return B;
}

} // namespace numbits
