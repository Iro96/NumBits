#pragma once
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace numbits {

/**
 * @brief Compute the sum of all elements in a numeric vector.
 *
 * This template function accepts a std::vector<T> where T must be an arithmetic type
 * (integral or floating‑point). It accumulates all elements into a larger accumulator
 * type (either long long or double) to reduce overflow/precision issues.
 *
 * @tparam T The element type of the input vector. Must satisfy std::is_arithmetic_v<T>.
 * @param data A vector of numeric elements to sum.
 * @return The sum of all elements, returned as the accumulator type (long long or double).
 * @note The function is marked noexcept since no exceptions are thrown inside the loop.
 * @throws None
 */
template <typename T>
inline auto sum(const std::vector<T>& data) noexcept {
    static_assert(std::is_arithmetic_v<T>, "sum() requires arithmetic type");
    using AccT = std::conditional_t<std::is_integral_v<T>, long long, double>;
    AccT total = 0;
    const T* ptr = data.data();
    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i)
        total += ptr[i];
    return total;
}

/**
 * @brief Compute the arithmetic mean (average) of the elements in a numeric vector.
 *
 * This function computes the mean by summing all elements of the vector via sum()
 * and then dividing by the number of elements. The vector must not be empty.
 *
 * @tparam T The element type of the input vector. Must satisfy std::is_arithmetic_v<T>.
 * @param data A vector of numeric elements whose mean is to be calculated.
 * @return The mean (as double) of the provided elements.
 * @throws std::domain_error If the input vector is empty.
 */
template <typename T>
inline double mean(const std::vector<T>& data) {
    if ( data.empty() )
        throw std::domain_error("mean: cannot compute mean of empty vector");
    return static_cast<double>( sum(data) ) / static_cast<double>( data.size() );
}

} // namespace numbits
