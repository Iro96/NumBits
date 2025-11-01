#pragma once
#include <vector>
#include <numeric>
#include <functional>
#include <limits>
#include <stdexcept>

namespace numbits {

/**
 * @brief Compute the total number of elements described by a shape.
 *
 * Calculates the product of all dimensions in `shape`, representing the total number
 * of elements for that multi-dimensional shape.
 *
 * @param shape Vector of dimension sizes (each element is a dimension length).
 * @return size_t Product of all elements in `shape`.
 * @throws std::overflow_error if the product would overflow `size_t`.
 */
inline size_t total_size(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        if (dim && total > std::numeric_limits<size_t>::max() / dim)
            throw std::overflow_error("total_size(): overflow in shape product");
        total *= dim;
    }
    return total;
}

} // namespace numbits