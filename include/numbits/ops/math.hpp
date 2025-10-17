#pragma once
#include "../core/ndarray.hpp"
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <functional>
#include <vector>

namespace numbits {

/**
 * @brief Apply a unary function elementwise to an n-dimensional ndarray.
 *
 * @tparam T Floating-point type.
 * @tparam Func Callable taking T and returning T.
 * @param A Input ndarray of any rank.
 * @param func Unary function to apply elementwise.
 * @param name Function name (for error messages).
 * @return ndarray<T> Result of applying func to each element of A.
 *
 * @throws std::logic_error if A has uninitialized data.
 */
template <typename T, typename Func>
ndarray<T> elementwise(const ndarray<T>& A, Func func, const char* name) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::elementwise requires floating-point T");

    const auto& shape = A.shape();
    if (A.size() == 0)
        throw std::logic_error(std::string(name) + ": input ndarray has no data");

    ndarray<T> B(shape);
    const size_t total = A.size();

    for (size_t i = 0; i < total; ++i)
        B[i] = func(A[i]);

    return B;
}

/**
 * @brief Elementwise exponential: B = exp(A)
 */
template <typename T>
ndarray<T> exp(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::exp(x); }, "exp");
}

/**
 * @brief Elementwise square root: B = sqrt(A)
 *
 * @throws std::domain_error if any element is negative.
 */
template <typename T>
ndarray<T> sqrt(const ndarray<T>& A) {
    return elementwise(A, [](T x){
        if (x < T(0)) throw std::domain_error("sqrt: negative input value");
        return std::sqrt(x);
    }, "sqrt");
}

/**
 * @brief Elementwise natural logarithm: B = log(A)
 *
 * @throws std::domain_error if any element is non-positive.
 */
template <typename T>
ndarray<T> log(const ndarray<T>& A) {
    return elementwise(A, [](T x){
        if (x <= T(0)) throw std::domain_error("log: input must be positive");
        return std::log(x);
    }, "log");
}

/**
 * @brief Elementwise sine: B = sin(A)
 */
template <typename T>
ndarray<T> sin(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::sin(x); }, "sin");
}

/**
 * @brief Elementwise cosine: B = cos(A)
 */
template <typename T>
ndarray<T> cos(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::cos(x); }, "cos");
}

/**
 * @brief Elementwise tangent: B = tan(A)
 */
template <typename T>
ndarray<T> tan(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::tan(x); }, "tan");
}

/**
 * @brief Elementwise power: B = A^exponent
 */
template <typename T>
ndarray<T> pow(const ndarray<T>& A, T exponent) {
    return elementwise(A, [exponent](T x){ return std::pow(x, exponent); }, "pow");
}

} // namespace numbits
