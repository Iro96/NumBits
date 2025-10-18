#pragma once
#include "../core/ndarray.hpp"
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <utility>

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
 */
template <typename T, typename Func>
ndarray<T> elementwise(const ndarray<T>& A, const Func& func, const char* name) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::elementwise requires floating-point T");
    using R = std::invoke_result_t<Func&, T>;
    static_assert(std::is_convertible_v<R, T>,
                  "numbits::elementwise: func(T) must return (or convert to) T");

    const auto& shape = A.shape();
    if (A.size() == 0)
        return ndarray<T>(shape);  // preserve shape for empty arrays

    ndarray<T> B(shape);
    for (size_t i = 0; i < A.size(); ++i)
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
 * Handles tiny negative values caused by floating-point errors.
 *
 * @param tol Optional tolerance for negative values considered as zero (default 1e-12).
 * @throws std::domain_error if any element is less than -tol.
 */
template <typename T>
ndarray<T> sqrt(const ndarray<T>& A, T tol = 1e-12) {
    return elementwise(A, [tol](T x){
        if (x < -tol)
            throw std::domain_error("sqrt: negative input value");
        return std::sqrt(x < T(0) ? T(0) : x); // clamp tiny negatives to zero
    }, "sqrt");
}

/**
 * @brief Elementwise natural logarithm: B = log(A)
 *
 * @param tol Optional tolerance for values considered as positive (default 1e-12).
 * @throws std::domain_error if any element is <= tol.
 */
template <typename T>
ndarray<T> log(const ndarray<T>& A, T tol = 1e-12) {
    return elementwise(A, [tol](T x){
        if (x <= tol)
            throw std::domain_error("log: input must be positive");
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
 * @brief Elementwise power with integer exponent (optimized)
 *
 * @tparam T Floating-point type.
 * @tparam I Integral exponent type.
 * @param A Input ndarray.
 * @param exponent Integer exponent.
 * @return ndarray<T> Result of raising each element of A to exponent.
 *
 * @throws std::domain_error if base is zero and exponent is negative.
 */
template <typename T, typename I,
          std::enable_if_t<std::is_integral_v<I>, int> = 0>
ndarray<T> pow(const ndarray<T>& A, I exponent) {
    return elementwise(A, [exp = exponent](T x) -> T {
        if (x == T(0) && exp < 0)
            throw std::domain_error("pow: zero cannot be raised to a negative exponent");

        T base = x, res = T(1);
        using U = std::make_unsigned_t<I>;
        U n = (exp < 0) ? static_cast<U>(-(exp + 1)) + 1 : static_cast<U>(exp);

        while (n) {
            if (n & 1) res *= base;
            base *= base;
            n >>= 1;
        }

        return (exp >= 0) ? res : T(1) / res;
    }, "pow");
}

/**
 * @brief Elementwise power with floating-point exponent
 *
 * @tparam T Floating-point type of the array.
 * @tparam F Floating-point exponent type.
 * @param A Input ndarray.
 * @param exponent Floating-point exponent.
 * @return ndarray<T> Result of raising each element of A to exponent.
 *
 * @throws std::domain_error if base is zero and exponent <= 0.
 */
template <typename T, typename F,
          std::enable_if_t<std::is_floating_point_v<F>, int> = 0>
ndarray<T> pow(const ndarray<T>& A, F exponent) {
    return elementwise(A, [exp = exponent](T x) -> T {
        if (x == T(0) && exp <= T(0))
            throw std::domain_error("pow: zero cannot be raised to non-positive exponent");
        return std::pow(x, exp);
    }, "pow");
}

} // namespace numbits
