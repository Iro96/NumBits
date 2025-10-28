#pragma once
#include <vector>
#include <numeric>
#include <functional>
#include <limits>
#include <stdexcept>

namespace numbits {

/**
 * @brief Compute the total number of elements implied by a shape vector.
 *
 * Multiplies all dimensions in `shape` to compute the total element count.
 * 
 * Example:
 * @code
 * numbits::total_size({3, 4});   // returns 12
 * numbits::total_size({2, 3, 0}); // returns 0 (empty axis)
 * @endcode
 *
 * @param shape A vector of dimension sizes (each typically >= 1).
 * @return The product of all dimensions, or 1 if `shape` is empty.
 *
 * @throws std::overflow_error If the product exceeds `std::numeric_limits<size_t>::max()`.
 *
 * @note This function detects overflow to avoid wrap-around when very large shapes are used.
 * @note An empty `shape` corresponds to a scalar or 0-D array and returns 1.
 */
constexpr size_t total_size(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        if (dim != 0 && total > std::numeric_limits<size_t>::max() / dim)
            throw std::overflow_error("total_size(): overflow in shape product");
        total *= dim;
    }
    return total;
}

} // namespace numbits
