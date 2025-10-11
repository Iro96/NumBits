#pragma once
#include "../core/ndarray.hpp"
#include "../ops/reduction.hpp"  // for mean()
#include <cmath>
#include <type_traits>

namespace numbits {

/**
 * @brief Compute the variance of elements in an ndarray.
 *
 * @tparam T Numeric type (usually float or double)
 * @param A Input ndarray
 * @return Variance of all elements
 */
template <typename T>
T variance(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "variance requires a numeric type");

    T m = mean(A);
    T var = 0;
    for (const auto& x : A.data())
        var += (x - m) * (x - m);
    return var / static_cast<T>(A.size());
}

/**
 * @brief Compute the standard deviation of an ndarray.
 *
 * @tparam T Numeric type (usually float or double)
 * @param A Input ndarray
 * @return Standard deviation
 */
template <typename T>
T stddev(const ndarray<T>& A) {
    return std::sqrt(variance(A));
}

} // namespace numbits
