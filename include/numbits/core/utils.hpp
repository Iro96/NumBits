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

} // namespace numbits
