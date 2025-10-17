#pragma once
#include <vector>
#include <numeric>
#include <algorithm>

namespace numbits {

/**
 * @brief Compute the sum of all elements in a vector.
 */
template <typename T>
T sum(const std::vector<T>& data) {
    return std::accumulate(data.begin(), data.end(), T(0));
}

/**
 * @brief Compute the mean (average) of all elements in a vector.
 *        Returns 0 if the vector is empty.
 */
template <typename T>
T mean(const std::vector<T>& data) {
    if (data.empty()) {
        return T(0);
    }
    return sum(data) / static_cast<T>(data.size());
}

/**
 * @brief Compute the total number of elements in an n-dimensional array.
 *
 * Given a vector of dimensions, returns the product of all dimensions.
 * Useful for allocating contiguous storage for ndarrays or validating
 * that a data vector matches the array shape.
 *
 * Example:
 * @code
 * std::vector<size_t> shape = {2, 3, 4};
 * size_t n = numbits::total_size(shape); // n == 24
 * @endcode
 *
 * @param dims Vector of sizes for each dimension
 * @return Total number of elements (product of dims)
 */
inline size_t total_size(const std::vector<size_t>& dims) {
    size_t s = 1;
    for (auto d : dims) s *= d;
    return s;
}

} // namespace numbits
