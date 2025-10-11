#pragma once
#include "../core/ndarray.hpp"
#include "../ops/reduction.hpp"  // for mean()
#include <cmath>
#include <type_traits>
#include <stdexcept>

namespace numbits {

/**
 * @brief Compute the variance of elements in an ndarray.
 *
 * Returns the population variance (not sample variance).
 * Always returns double for precision, even if T is integral.
 *
 * @tparam T Arithmetic type (integer or floating-point)
 * @param A Input ndarray
 * @return double Variance of elements
 * @throws std::invalid_argument if A is empty
 */
template <typename T>
double variance(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "numbits::variance requires numeric T");

    const auto count = A.size();
    if (count == 0)
        throw std::invalid_argument("variance: ndarray must contain at least one element");

    const double m = static_cast<double>(mean(A));
    double var = 0.0;
    for (const auto& x : A.data()) {
        const double d = static_cast<double>(x) - m;
        var += d * d;
    }
    return var / static_cast<double>(count);
}

/**
 * @brief Compute the standard deviation of elements in an ndarray.
 *
 * Returns double precision regardless of input type.
 *
 * @tparam T Arithmetic type
 * @param A Input ndarray
 * @return double Standard deviation
 */
template <typename T>
double stddev(const ndarray<T>& A) {
    return std::sqrt(variance<T>(A));
}

} // namespace numbits
