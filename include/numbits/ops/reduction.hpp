#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>
#include <type_traits>
#include <limits>

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
 *
 * Performs integer division (truncates fractional part).  
 * Uses a widened accumulator to prevent overflow, and checks
 * that the result fits into T before narrowing.
 *
 * @tparam T Integral type.
 * @throws std::domain_error if A.size() == 0
 * @throws std::overflow_error if the mean cannot fit into T
 */
template <typename T>
std::enable_if_t<std::is_integral_v<T>, T>
mean_truncated(const ndarray<T>& A) {
    if (A.size() == 0)
        throw std::domain_error("mean_truncated: cannot compute mean of empty ndarray");

    using Wide = long long;  // wide enough for most integer accumulation
    static_assert(std::numeric_limits<Wide>::digits >= std::numeric_limits<T>::digits,
                  "mean_truncated: accumulator type too small");

    Wide total = 0;
    for (const auto& v : A.data())
        total += static_cast<Wide>(v);

    Wide divisor = static_cast<Wide>(A.size());
    Wide quotient = total / divisor;

    if (quotient > static_cast<Wide>(std::numeric_limits<T>::max()) ||
        quotient < static_cast<Wide>(std::numeric_limits<T>::min())) {
        throw std::overflow_error("mean_truncated: result out of range for target type");
    }

    return static_cast<T>(quotient);
}

} // namespace numbits