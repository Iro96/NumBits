/**
 * @file ndarray_manipulation.hpp
 * @brief Array manipulation operations (concatenate, stack, split, tile, reshape_advanced).
 *
 * Provides functions to manipulate array structure:
 *   - concatenate: Join arrays along existing axis
 *   - stack: Join arrays along new axis
 *   - split: Divide array into subarrays
 *   - tile: Repeat array elements
 *   - hstack, vstack, dstack: Convenience stacking functions
 *
 * @namespace numbits
 */

#pragma once

#include "ndarray.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace numbits {

/**
 * @brief Concatenate multiple ndarrays along an existing axis.
 *
 * Works similarly to NumPy's `np.concatenate`. All arrays must:
 *   - Have the same number of dimensions
 *   - Match in all dimensions except the specified axis
 *
 * @tparam T Element data type
 * @param ndarrays Vector of ndarrays to concatenate
 * @param axis Axis along which concatenation occurs (0-based)
 *
 * @return ndarray<T> A new array containing the concatenated result
 *
 * @throws std::runtime_error If:
 *   - Input list is empty
 *   - Arrays have mismatched ranks
 *   - Dimensions (other than axis) do not match
 *   - Axis is out of range
 */
template<typename T>
ndarray<T> concatenate(const std::vector<ndarray<T>>& ndarrays, size_t axis = 0) {
    if (ndarrays.empty()) {
        throw std::runtime_error("Cannot concatenate empty ndarray list");
    }
    
    if (ndarrays.size() == 1) {
        return ndarrays[0];
    }
    
    // Verify all ndarrays have same number of dimensions
    size_t ndim = ndarrays[0].ndim();
    for (const auto& arr : ndarrays) {
        if (arr.ndim() != ndim) {
            throw std::runtime_error("All ndarrays must have same number of dimensions");
        }
    }
    
    if (axis >= ndim) {
        throw std::runtime_error("Axis out of range");
    }
    
    // Verify all dimensions except axis are the same
    Shape result_shape = ndarrays[0].shape();
    size_t total_axis_size = 0;
    for (const auto& arr : ndarrays) {
        for (size_t i = 0; i < ndim; ++i) {
            if (i != axis && arr.shape()[i] != result_shape[i]) {
                throw std::runtime_error("ndarray dimensions incompatible for concatenation");
            }
        }
        total_axis_size += arr.shape()[axis];
    }
    result_shape[axis] = total_axis_size;
    
    // Create result ndarray
    ndarray<T> result(result_shape);
    
    // Copy data
    size_t result_offset = 0;
    for (const auto& arr : ndarrays) {
        size_t axis_size = arr.shape()[axis];
        
        // Copy each element
        for (size_t i = 0; i < arr.size(); ++i) {
            // Calculate position in source ndarray
            std::vector<size_t> src_indices = unravel_index(i, arr.shape(), arr.strides());
            
            // Calculate position in result ndarray (adjust axis)
            std::vector<size_t> dst_indices = src_indices;
            dst_indices[axis] += result_offset;
            
            // Copy element
            size_t dst_idx = flatten_index(dst_indices, result.strides());
            if (dst_idx < result.size()) {
                result[dst_idx] = arr[i];
            }
        }
        result_offset += axis_size;
    }
    
    return result;
}

/**
 * @brief Stack multiple ndarrays along a new axis.
 *
 * Equivalent to NumPy's `np.stack`. Requirements:
 *   - All arrays must have the *same shape*
 *   - A new axis is inserted before indexing
 *
 * Example: stacking 3 arrays of shape (2,3) along axis=0  
 * Result shape = (3,2,3)
 *
 * @tparam T Element type
 * @param ndarrays Vector of ndarrays to stack
 * @param axis Position of the new axis (0 ≤ axis ≤ ndim)
 *
 * @return ndarray<T> Stacked result array
 *
 * @throws std::runtime_error If:
 *   - Input is empty
 *   - Arrays have mismatched shapes
 */
template<typename T>
ndarray<T> stack(const std::vector<ndarray<T>>& ndarrays, size_t axis = 0) {
    if (ndarrays.empty()) {
        throw std::runtime_error("Cannot stack empty ndarray list");
    }
    
    // All ndarrays must have the same shape
    Shape base_shape = ndarrays[0].shape();
    for (const auto& arr : ndarrays) {
        if (arr.shape() != base_shape) {
            throw std::runtime_error("All ndarrays must have the same shape for stacking");
        }
    }
    
    // Insert new axis
    Shape result_shape;
    result_shape.reserve(base_shape.size() + 1);
    for (size_t i = 0; i < axis; ++i) {
        result_shape.push_back(base_shape[i]);
    }
    result_shape.push_back(ndarrays.size());
    for (size_t i = axis; i < base_shape.size(); ++i) {
        result_shape.push_back(base_shape[i]);
    }
    
    ndarray<T> result(result_shape);
    
    // Copy data
    size_t elements_per_ndarray = ndarrays[0].size();
    
    for (size_t arr_idx = 0; arr_idx < ndarrays.size(); ++arr_idx) {
        const auto& arr = ndarrays[arr_idx];
        for (size_t i = 0; i < elements_per_ndarray; ++i) {
            // Get indices in source ndarray
            std::vector<size_t> src_indices = unravel_index(i, arr.shape(), arr.strides());
            
            // Insert new axis index
            std::vector<size_t> dst_indices = src_indices;
            dst_indices.insert(dst_indices.begin() + axis, arr_idx);
            
            // Calculate destination index
            size_t result_idx = flatten_index(dst_indices, result.strides());
            if (result_idx < result.size()) {
                result[result_idx] = arr[i];
            }
        }
    }
    
    return result;
}

/**
 * @brief Split an ndarray into multiple subarrays along a given axis.
 *
 * Behaves similarly to `np.split`.  
 * The array is split into segments defined by `indices`, representing boundaries.
 *
 * Example:  
 * shape = (10, 3), axis=0, indices = {2, 5}  
 * Result → subarrays of sizes: [0–2], [2–5], [5–10]
 *
 * @tparam T Element type
 * @param arr Input ndarray
 * @param axis Axis along which the split occurs
 * @param indices A list of split boundaries (exclusive)
 *
 * @return std::vector<ndarray<T>> List of resulting subarrays
 *
 * @throws std::runtime_error If:
 *   - Axis is out of range
 */
template<typename T>
std::vector<ndarray<T>> split(const ndarray<T>& arr, size_t axis, const std::vector<size_t>& indices) {
    if (axis >= arr.ndim()) {
        throw std::runtime_error("Axis out of range");
    }
    
    std::vector<size_t> split_points = indices;
    split_points.insert(split_points.begin(), 0);
    split_points.push_back(arr.shape()[axis]);
    
    std::vector<ndarray<T>> results;
    for (size_t i = 0; i < split_points.size() - 1; ++i) {
        size_t start = split_points[i];
        size_t end = split_points[i + 1];
        size_t length = end - start;
        
        Shape result_shape = arr.shape();
        result_shape[axis] = length;
        ndarray<T> result(result_shape);
        
        // Copy slice
        size_t copy_size = compute_size(result_shape);
        Strides result_strides = compute_strides(result_shape);
        
        for (size_t j = 0; j < copy_size; ++j) {
            // Get indices in result ndarray
            std::vector<size_t> dst_indices = unravel_index(j, result_shape, result_strides);
            
            // Calculate corresponding indices in source ndarray
            std::vector<size_t> src_indices = dst_indices;
            src_indices[axis] += start;
            
            // Copy element
            size_t src_idx = flatten_index(src_indices, arr.strides());
            if (src_idx < arr.size() && j < result.size()) {
                result[j] = arr[src_idx];
            }
        }
        
        results.push_back(std::move(result));
    }
    
    return results;
}

/**
 * @brief Repeat elements of an ndarray along a given axis.
 *
 * Equivalent to NumPy's `np.repeat` (but repeats entire slices along the axis).
 *
 * Example:  
 * arr.shape = (2,3), repeat = 2, axis = 0  
 * Result.shape = (4,3)
 *
 * @tparam T Element type
 * @param arr Input ndarray
 * @param repeats Number of repetitions
 * @param axis Axis along which to repeat
 *
 * @return ndarray<T> Repeated ndarray
 *
 * @throws std::runtime_error If axis is out of range
 */
template<typename T>
ndarray<T> repeat(const ndarray<T>& arr, size_t repeats, size_t axis = 0) {
    if (axis >= arr.ndim()) {
        throw std::runtime_error("Axis out of range");
    }
    
    Shape result_shape = arr.shape();
    result_shape[axis] *= repeats;
    ndarray<T> result(result_shape);
    
    size_t axis_size = arr.shape()[axis];
    
    for (size_t r = 0; r < repeats; ++r) {
        for (size_t i = 0; i < arr.size(); ++i) {
            // Get source indices
            std::vector<size_t> src_indices = unravel_index(i, arr.shape(), arr.strides());
            
            // Calculate destination indices (offset along axis)
            std::vector<size_t> dst_indices = src_indices;
            dst_indices[axis] = src_indices[axis] + r * axis_size;
            
            // Copy element
            size_t result_idx = flatten_index(dst_indices, result.strides());
            if (result_idx < result.size()) {
                result[result_idx] = arr[i];
            }
        }
    }
    
    return result;
}

/**
 * @brief Tile an ndarray by repeating it across each dimension.
 *
 * Equivalent to NumPy's `np.tile`.  
 * `reps` must match the number of dimensions of the array.
 *
 * Example:  
 * arr.shape = (2, 1), reps = {2, 3}  
 * Result.shape = (4, 3)
 *
 * @tparam T Element type
 * @param arr Input ndarray
 * @param reps Vector specifying repetitions along each axis
 *
 * @return ndarray<T> Tiled ndarray
 *
 * @throws std::runtime_error If reps.size() != arr.ndim()
 */
template<typename T>
ndarray<T> tile(const ndarray<T>& arr, const std::vector<size_t>& reps) {
    if (reps.size() != arr.ndim()) {
        throw std::runtime_error("Number of repetitions must match number of dimensions");
    }
    
    Shape result_shape = arr.shape();
    for (size_t i = 0; i < reps.size(); ++i) {
        result_shape[i] *= reps[i];
    }
    
    ndarray<T> result(result_shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        std::vector<size_t> result_indices = unravel_index(i, result_shape, result.strides());
        std::vector<size_t> arr_indices = result_indices;
        for (size_t j = 0; j < arr_indices.size(); ++j) {
            arr_indices[j] %= arr.shape()[j];
        }
        size_t arr_idx = flatten_index(arr_indices, arr.strides());
        result[i] = arr[arr_idx];
    }
    
    return result;
}

} // namespace numbits

