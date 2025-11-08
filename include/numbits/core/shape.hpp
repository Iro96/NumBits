#pragma once
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits>

namespace numbits {

/**
 * @brief Compute the total number of elements implied by a shape vector.
 *
 * Given a vector of dimension extents (each entry &gt;= 1), this function
 * returns the product of all entries, representing the total number
 * of elements in an array of that shape.
 *
 * @param shape A std::vector<size_t> containing the extents of each dimension.
 *              Each element may be zero (though zero dims will lead to total = 0).
 * @return The product of all entries in `shape`.
 *
 * @throws std::overflow_error
 *         If the multiplication would overflow the range of size_t
 *         (i.e., if total > max(size_t)/dim at any step).
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
