#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

namespace numbits {

template <typename T>
/**
 * @brief Compute the sum of all elements in a vector.
 *
 * Template parameter `T` must be an arithmetic type; a compile-time error occurs otherwise.
 *
 * @tparam T Element type of the input vector; must satisfy std::is_arithmetic.
 * @param data Vector of values to sum.
 * @return Accumulator containing the sum: `long long` when `T` is an integral type, `double` when `T` is a floating-point type.
 */
inline auto sum(const std::vector<T>& data) noexcept {
    static_assert(std::is_arithmetic_v<T>, "sum() requires arithmetic type");
    using AccT = std::conditional_t<std::is_integral_v<T>, long long, double>;
    AccT total = 0;
    const T* ptr = data.data();
    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) total += ptr[i];
    return total;
}

template <typename T>
/**
 * @brief Computes the arithmetic mean of the values in a vector.
 *
 * @tparam T Element type; must be an arithmetic type (integral or floating-point).
 * @param data Input values to average. Must not be empty.
 * @return double The arithmetic mean of the elements in `data`.
 * @throws std::domain_error If `data` is empty.
 */
double mean(const std::vector<T>& data) {
    if (data.empty())
        throw std::domain_error("mean: cannot compute mean of empty vector");
    return static_cast<double>(sum(data)) / static_cast<double>(data.size());
}

} // namespace numbits