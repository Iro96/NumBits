#pragma once
#include <vector>
#include <numeric>
#include <functional>

namespace numbits {

/**
 * @brief Compute the total number of elements for a given shape vector.
 * 
 * Example:
 *     total_size({3, 4}) == 12
 * 
 * @param shape A vector of dimension sizes.
 * @return size_t The product of all dimensions (1 for empty shape).
 */
constexpr size_t total_size(const std::vector<size_t>& shape) noexcept {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
}

} // namespace numbits
