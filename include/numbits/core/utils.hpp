#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>

namespace numbits {

/**
 * @brief Computes the sum of all elements in a numeric vector.
 *
 * This function iterates through all elements in the input vector and returns
 * their accumulated sum. The accumulation type is promoted automatically:
 * - For integral types (`int`, `long`, etc.), accumulation is done using `long long`.
 * - For floating-point types (`float`, `double`), accumulation is done using `double`.
 *
 * The function requires that the element type `T` satisfies `std::is_arithmetic_v<T>`.
 *
 * Example:
 * @code
 * std::vector<int> v = {1, 2, 3, 4};
 * auto total = numbits::sum(v); // total == 10 (type: long long)
 *
 * std::vector<double> d = {1.5, 2.5};
 * auto total_d = numbits::sum(d); // total_d == 4.0 (type: double)
 * @endcode
 *
 * @tparam T Element type of the vector (must be arithmetic).
 * @param data Input vector of numeric values.
 * @return The total sum of elements, promoted to `long long` or `double` depending on `T`.
 *
 * @throws std::logic_error Never throws (no runtime exceptions).
 */
template <typename T>
auto sum(const std::vector<T>& data) {
    static_assert(std::is_arithmetic_v<T>, "sum() requires arithmetic type");
    using AccT = std::conditional_t<std::is_integral_v<T>, long long, double>;
    return std::accumulate(data.begin(), data.end(), AccT(0));
}

/**
 * @brief Computes the arithmetic mean (average) of all elements in a numeric vector.
 *
 * The mean is defined as the sum of all elements divided by the number of elements.
 * The result is returned as a `double`, ensuring floating-point precision even
 * when the input type `T` is integral.
 *
 * If the input vector is empty, the function returns `0.0` to avoid division by zero.
 *
 * Example:
 * @code
 * std::vector<int> v = {1, 2, 3, 4};
 * double avg = numbits::mean(v); // avg == 2.5
 *
 * std::vector<double> d = {1.2, 3.8, 5.0};
 * double avg_d = numbits::mean(d); // avg_d == 3.333333...
 *
 * std::vector<int> empty;
 * double avg_empty = numbits::mean(empty); // avg_empty == 0.0
 * @endcode
 *
 * @tparam T Element type of the vector (must be arithmetic).
 * @param data Input vector of numeric values.
 * @return The mean (average) of the input values as a double.
 *
 * @note This function performs floating-point division even for integer vectors.
 * @note For empty vectors, returns `0.0` instead of throwing an exception.
 *
 * @throws std::logic_error Never throws (no runtime exceptions).
 */
template <typename T>
double mean(const std::vector<T>& data) {
    if (data.empty()) return 0.0;
    return static_cast<double>(sum(data)) / static_cast<double>(data.size());
}

} // namespace numbits
