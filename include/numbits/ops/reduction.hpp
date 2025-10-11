#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>
#include <type_traits>

namespace numbits {

/**
 * @brief Compute sum of all elements in an ndarray.
 */
template <typename T>
T sum(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "numbits::sum requires arithmetic T");

    T s = T(0);
    for (const auto& v : A.data())
        s += v;

    return s;
}

/**
 * @brief Compute mean of all elements in an ndarray.
 *        Always returns double for precision safety.
 * @throws std::domain_error if A.size() == 0
 */
template <typename T>
double mean(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "numbits::mean requires arithmetic T");

    if (A.size() == 0)
        throw std::domain_error("mean: cannot compute mean of empty ndarray");

    return static_cast<double>(sum(A)) /
           static_cast<double>(A.size());
}

/**
 * @brief Compute mean of all elements in an ndarray, truncating to integer result.
 *        Useful only when truncation is explicitly desired.
 * @note Performs integer division (truncates fractional part).
 */
template <typename T>
std::enable_if_t<std::is_integral_v<T>, T>
mean_truncated(const ndarray<T>& A) {
    if (A.size() == 0)
        throw std::domain_error("mean_truncated: cannot compute mean of empty ndarray");

    return sum(A) / static_cast<T>(A.size()); // integer division
}

} // namespace numbits
