/**
 * @file utils.hpp
 * @brief Utility functions for array shape, strides, and index calculations.
 *
 * This header provides core helper functions for n-dimensional array manipulation:
 *   - `compute_size`: Calculate total number of elements from a shape
 *   - `compute_strides`: Compute memory strides for efficient indexing
 *   - `flatten_index`: Convert multi-dimensional indices to a flat index
 *   - `unravel_index`: Convert flat index back to multi-dimensional indices
 *   - `broadcast_shapes`: Determine the resulting shape when broadcasting two arrays
 *   - `can_broadcast`: Check if two shapes are broadcast-compatible
 *   - `shape_to_string`: Format shape as a human-readable string
 *
 * @namespace numbits
 */

#pragma once

#include "types.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <string>
#include <stdexcept>

namespace numbits {

/**
 * @brief Compute the total number of elements in an array given its shape.
 *
 * Example:
 * @code
 * Shape s = {3, 4, 5};
 * size_t total = compute_size(s); // returns 60
 * @endcode
 *
 * @param shape The shape of the array
 * @return Total number of elements
 */
inline size_t compute_size(const Shape& shape) {
    if (shape.empty()) return 1;
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

/**
 * @brief Compute memory strides for a given shape.
 *
 * Strides indicate the number of elements to skip in memory to move along each dimension.
 * For a contiguous row-major array:
 * - The last dimension stride is 1
 * - Strides are cumulative products of the following dimensions
 *
 * Example:
 * @code
 * Shape s = {3, 4, 5};
 * Strides st = compute_strides(s); // returns {20, 5, 1}
 * @endcode
 *
 * @param shape The shape of the array
 * @return Strides corresponding to each dimension
 */
inline Strides compute_strides(const Shape& shape) {
    if (shape.empty()) return {};
    Strides strides(shape.size());
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

/**
 * @brief Convert multi-dimensional indices into a flat (1D) index.
 *
 * Example:
 * @code
 * Strides st = {20, 5, 1};
 * std::vector<size_t> idx = {2, 1, 3};
 * size_t flat = flatten_index(idx, st); // returns 2*20 + 1*5 + 3*1 = 48
 * @endcode
 *
 * @param indices Multi-dimensional indices
 * @param strides Strides of the array
 * @return Flattened 1D index
 */
inline size_t flatten_index(const std::vector<size_t>& indices, const Strides& strides) {
    size_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

/**
 * @brief Convert a flat index back to multi-dimensional indices.
 *
 * Example:
 * @code
 * Shape s = {3, 4, 5};
 * Strides st = compute_strides(s);
 * std::vector<size_t> idx = unravel_index(48, s, st); // returns {2, 1, 3}
 * @endcode
 *
 * @param flat_idx Flat 1D index
 * @param shape Shape of the array
 * @param strides Strides of the array
 * @return Multi-dimensional indices corresponding to the flat index
 */
inline std::vector<size_t> unravel_index(size_t flat_idx, const Shape& shape, const Strides& strides) {
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        indices[i] = flat_idx / strides[i];
        flat_idx %= strides[i];
    }
    return indices;
}

/**
 * @brief Compute the broadcasted shape of two arrays.
 *
 * Broadcasting rules:
 * - Dimensions are compatible if equal or one of them is 1
 * - Output shape has the maximum along each dimension
 *
 * Example:
 * @code
 * Shape s1 = {3, 1, 5};
 * Shape s2 = {1, 4, 5};
 * Shape result = broadcast_shapes(s1, s2); // returns {3, 4, 5}
 * @endcode
 *
 * @param shape1 First array shape
 * @param shape2 Second array shape
 * @return Broadcasted shape
 * @throws std::runtime_error if shapes cannot be broadcasted
 */
inline Shape broadcast_shapes(const Shape& shape1, const Shape& shape2) {
    size_t ndim = std::max(shape1.size(), shape2.size());
    Shape result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::runtime_error("Cannot broadcast shapes");
        }

        result[ndim - 1 - i] = std::max(dim1, dim2);
    }

    return result;
}

/**
 * @brief Check whether two shapes are broadcast-compatible.
 *
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return true if shapes can be broadcasted, false otherwise
 */
inline bool can_broadcast(const Shape& shape1, const Shape& shape2) {
    try {
        broadcast_shapes(shape1, shape2);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Convert a shape to a human-readable string.
 *
 * Example:
 * @code
 * Shape s = {3, 4, 5};
 * std::string str = shape_to_string(s); // "(3, 4, 5)"
 * @endcode
 *
 * @param shape The shape to format
 * @return String representation of the shape
 */
inline std::string shape_to_string(const Shape& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    if (shape.size() == 1) oss << ",";
    oss << ")";
    return oss.str();
}

} // namespace numbits
