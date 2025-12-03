/**
 * @file indexing.hpp
 * @brief Advanced indexing and slicing operations.
 *
 * Provides:
 *   - Slice specification for range-based indexing
 *   - take(): Extract elements at specified indices along an axis
 *   - Boolean indexing via where()
 *   - Advanced indexing with index arrays
 *   - Basic 1D slicing
 *
 * @namespace numbits
 */

#pragma once

#include "ndarray.hpp"
#include "broadcasting.hpp"
#include <vector>
#include <stdexcept>

namespace numbits {

/**
 * @struct Slice
 * @brief Represents a slice specification for range-based indexing.
 *
 * A Slice consists of:
 *  - `start`: starting index of the slice (inclusive)
 *  - `stop`: ending index of the slice (exclusive)
 *  - `step`: step size between elements
 *
 * A special case is `Slice::all()` which produces the slice (0, 0, 1),
 * interpreted by higher-level utilities as selecting the entire dimension.
 */
struct Slice {
    size_t start; ///< Starting index (inclusive)
    size_t stop;  ///< Ending index (exclusive)
    size_t step;  ///< Step between indices

    /**
     * @brief Construct a new Slice.
     * @param s Start index
     * @param e Stop index
     * @param st Step size
     */
    Slice(size_t s = 0, size_t e = 0, size_t st = 1) 
        : start(s), stop(e), step(st) {}

    /**
     * @brief Create a slice that selects the entire dimension.
     * @return Slice (0, 0, 1)
     */
    static Slice all() { return Slice(0, 0, 1); }
};

/**
 * @brief Extract elements of an ndarray along a given axis using explicit indices.
 *
 * Semantics:
 *  - Matches NumPy's `take` behavior, but simplified.
 *  - Replaces the selected axis with an axis of size `indices.size()`.
 *  - Performs bounds checking on indices.
 *
 * @tparam T Element type
 * @param arr The input array to extract from
 * @param indices A list of indices to select along the given axis
 * @param axis The axis along which to index (default: 0)
 * @return ndarray<T> A new array containing the gathered elements
 *
 * @throws std::runtime_error If axis is out of range
 * @throws std::out_of_range If any index is invalid
 */
template<typename T>
ndarray<T> take(const ndarray<T>& arr, const std::vector<size_t>& indices, size_t axis = 0) {
    if (axis >= arr.ndim()) {
        throw std::runtime_error("Axis out of range");
    }
    
    Shape result_shape = arr.shape();
    result_shape[axis] = indices.size();
    ndarray<T> result(result_shape);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (idx >= arr.shape()[axis]) {
            throw std::out_of_range("Index out of range");
        }
        
        // Copy slice at this index
        for (size_t j = 0; j < result.size() / indices.size(); ++j) {
            std::vector<size_t> result_indices = unravel_index(
                i * (result.size() / indices.size()) + j, 
                result_shape, result.strides()
            );

            std::vector<size_t> arr_indices = result_indices;
            arr_indices[axis] = idx;

            size_t arr_idx = flatten_index(arr_indices, arr.strides());
            size_t result_idx = flatten_index(result_indices, result.strides());

            result[result_idx] = arr[arr_idx];
        }
    }
    
    return result;
}

/**
 * @brief Elementwise selection based on a boolean condition array.
 *
 * Matches NumPy semantics:
 *   result[i] = condition[i] ? x[i] : y[i]
 *
 * All inputs are broadcast to a common shape using NumPy broadcasting rules.
 *
 * @tparam T Element type
 * @param condition Boolean ndarray controlling selection
 * @param x Values chosen where condition is true
 * @param y Values chosen where condition is false
 * @return ndarray<T> An array containing elementwise selections
 *
 * @throws std::runtime_error If shapes cannot be broadcast
 */
template<typename T>
ndarray<T> where(const ndarray<bool>& condition, const ndarray<T>& x, const ndarray<T>& y) {
    Shape xy_shape = broadcast_shapes(x.shape(), y.shape());
    Shape broadcast_shape = broadcast_shapes(condition.shape(), xy_shape);
    
    ndarray<bool> cond_broadcast = broadcast_to(condition, broadcast_shape);
    ndarray<T> x_broadcast = broadcast_to(x, broadcast_shape);
    ndarray<T> y_broadcast = broadcast_to(y, broadcast_shape);
    
    ndarray<T> result(broadcast_shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = cond_broadcast[i] ? x_broadcast[i] : y_broadcast[i];
    }
    
    return result;
}

/**
 * @brief Advanced indexing using per-dimension index arrays.
 *
 * Equivalent to NumPy's "advanced indexing" where each dimension receives
 * an array of integer indices, and the output is formed by selecting
 * arr[idx0[i], idx1[i], ..., idxN[i]] for each i.
 *
 * Requirements:
 *  - `indices.size() == arr.ndim()`
 *  - All index arrays must have the same length
 *
 * @tparam T Element type
 * @param arr Input array
 * @param indices A list of vectors, one per dimension
 * @return ndarray<T> A 1D array containing the selected elements
 *
 * @throws std::runtime_error If number of index arrays mismatches ndim
 * @throws std::runtime_error If index arrays have mismatched sizes
 */
template<typename T>
ndarray<T> advanced_indexing(const ndarray<T>& arr, 
                             const std::vector<std::vector<size_t>>& indices) {
    if (indices.size() != arr.ndim()) {
        throw std::runtime_error("Number of index ndarrays must match number of dimensions");
    }
    
    if (indices.empty()) {
        return ndarray<T>();
    }
    
    size_t result_size = indices[0].size();
    for (const auto& idx_arr : indices) {
        if (idx_arr.size() != result_size) {
            throw std::runtime_error("All index ndarrays must have the same size");
        }
    }
    
    ndarray<T> result({result_size});
    
    for (size_t i = 0; i < result_size; ++i) {
        std::vector<size_t> coords;
        coords.reserve(indices.size());

        for (const auto& idx_arr : indices) {
            coords.push_back(idx_arr[i]);
        }

        result[i] = arr.at(coords);
    }
    
    return result;
}

/**
 * @brief Perform simple slicing on a 1D ndarray.
 *
 * Equivalent to Python/NumPy slicing:
 *   arr[start:stop:step]
 *
 * Behavior:
 *  - Clamps `stop` to array size
 *  - Returns empty array if `start >= stop`
 *
 * @tparam T Element type
 * @param arr 1D input array
 * @param start Starting index (inclusive)
 * @param stop Ending index (exclusive)
 * @param step Step size (default 1)
 * @return ndarray<T> A sliced 1D array
 *
 * @throws std::runtime_error If arr is not 1D
 */
template<typename T>
ndarray<T> slice_1d(const ndarray<T>& arr, size_t start, size_t stop, size_t step = 1) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("slice_1d requires 1D ndarray");
    }
    
    if (stop > arr.size()) {
        stop = arr.size();
    }
    
    if (start >= stop) {
        return ndarray<T>({0});
    }
    
    size_t result_size = (stop - start + step - 1) / step;
    ndarray<T> result({result_size});
    
    for (size_t i = 0; i < result_size; ++i) {
        size_t idx = start + i * step;
        if (idx >= stop) break;
        result[i] = arr[idx];
    }
    
    return result;
}

} // namespace numbits
