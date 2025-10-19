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
 *         If A is empty (size 0), returns an empty ndarray with the same shape.
 *
 * @note This preserves the shape of empty arrays, allowing consistent
 *       behavior with broadcasting and shape-dependent operations.
 */
template <typename T, typename Func>
ndarray<T> elementwise(const ndarray<T>& A, Func&& func) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::elementwise requires floating-point T");
    using F = std::remove_reference_t<Func>;
    using R = std::invoke_result_t<F&, T>;
    static_assert(std::is_convertible_v<R, T>,
                  "numbits::elementwise: func(T) must return (or convert to) T");

    const auto& shape = A.shape();
    if (A.size() == 0)
        return ndarray<T>(shape);  // preserve shape for empty arrays

    ndarray<T> B(shape);
    auto&& f = std::forward<Func>(func);
    const size_t n = A.size();
    for (size_t i = 0; i < n; ++i)
        B[i] = f(A[i]);

    return B;
}

/**
 * @brief Elementwise exponential: B = exp(A)
 */
template <typename T>
ndarray<T> exp(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::exp(x); });
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
    const char* name = "sqrt";
    if (tol < T(0)) throw std::invalid_argument(std::string(name) + ": tol must be non-negative");
    return elementwise(A, [tol, name](T x){
        if (x < -tol)
            throw std::domain_error(std::string(name) + ": negative input value");
        return std::sqrt(x < T(0) ? T(0) : x); // clamp tiny negatives to zero
    });
}

/**
 * @brief Elementwise natural logarithm: B = log(A)
 *
 * @param tol Optional tolerance below which values are treated as non‑positive (default 1e-12).
 * @throws std::domain_error if any element is <= tol (i.e., requires x > tol).
 */
template <typename T>
ndarray<T> log(const ndarray<T>& A, T tol = 1e-12) {
    const char* name = "log";
    if (tol < T(0)) throw std::invalid_argument(std::string(name) + ": tol must be non-negative");
    return elementwise(A, [tol, name](T x){
        if (x <= tol)
            throw std::domain_error(std::string(name) + ": input must be positive");
        return std::log(x);
    });
}

/**
 * @brief Elementwise sine: B = sin(A)
 */
template <typename T>
ndarray<T> sin(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::sin(x); });
}

/**
 * @brief Elementwise cosine: B = cos(A)
 */
template <typename T>
ndarray<T> cos(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::cos(x); });
}

/**
 * @brief Elementwise tangent: B = tan(A)
 */
template <typename T>
ndarray<T> tan(const ndarray<T>& A) {
    return elementwise(A, [](T x){ return std::tan(x); });
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
    const char* name = "pow";
    return elementwise(A, [exp = exponent, name](T x) -> T {
        if (x == T(0) && exp < 0)
            throw std::domain_error(std::string(name) + ": zero cannot be raised to a negative exponent");

        T base = x, res = T(1);
        using U = std::make_unsigned_t<I>;
        // Convert negative exponent to unsigned for binary exponentiation.
        // The expression -(exp + 1) + 1 avoids undefined behavior when exp == INT_MIN,
        // because directly negating INT_MIN is undefined in C++.
        U n = (exp < 0) ? static_cast<U>(-(exp + 1)) + 1 : static_cast<U>(exp);

        while (n) {
            if (n & 1) res *= base;
            base *= base;
            n >>= 1;
        }

        return (exp >= 0) ? res : T(1) / res;
    });
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
    const char* name = "pow";
    return elementwise(A, [exp = exponent, name](T x) -> T {
        if (x == T(0) && exp <= T(0))
            throw std::domain_error(std::string(name) + ": zero cannot be raised to a negative exponent");
        return std::pow(x, exp);
    });
}

} // namespace numbits
