#pragma once

#include "ndarray.hpp"
#include <vector>
#include <stdexcept>

namespace numbits {

// Slice specification
struct Slice {
    size_t start;
    size_t stop;
    size_t step;
    
    Slice(size_t s = 0, size_t e = 0, size_t st = 1) 
        : start(s), stop(e), step(st) {}
    
    static Slice all() { return Slice(0, 0, 1); } // 0,0 means "all"
};

// Extract subndarray using indices
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
        // Simplified: iterate through all elements
        for (size_t j = 0; j < result.size() / indices.size(); ++j) {
            // Calculate source and destination indices
            std::vector<size_t> result_indices = unravel_index(i * (result.size() / indices.size()) + j, 
                                                                 result_shape, result.strides());
            std::vector<size_t> arr_indices = result_indices;
            arr_indices[axis] = idx;
            size_t arr_idx = flatten_index(arr_indices, arr.strides());
            size_t result_idx = flatten_index(result_indices, result.strides());
            result[result_idx] = arr[arr_idx];
        }
    }
    
    return result;
}

// Boolean indexing
template<typename T>
ndarray<T> where(const ndarray<bool>& condition, const ndarray<T>& x, const ndarray<T>& y) {
    if (x.shape() != y.shape()) {
        throw std::runtime_error("x and y must have the same shape");
    }
    
    Shape broadcast_shape = broadcast_shapes(condition.shape(), x.shape());
    ndarray<bool> cond_broadcast = broadcast_to(condition, broadcast_shape);
    ndarray<T> x_broadcast = broadcast_to(x, broadcast_shape);
    ndarray<T> y_broadcast = broadcast_to(y, broadcast_shape);
    
    ndarray<T> result(broadcast_shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = cond_broadcast[i] ? x_broadcast[i] : y_broadcast[i];
    }
    
    return result;
}

// Advanced indexing helper
template<typename T>
ndarray<T> advanced_indexing(const ndarray<T>& arr, 
                          const std::vector<std::vector<size_t>>& indices) {
    if (indices.size() != arr.ndim()) {
        throw std::runtime_error("Number of index ndarrays must match number of dimensions");
    }
    
    // For simplicity, assume all index ndarrays have the same size
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
        for (const auto& idx_arr : indices) {
            coords.push_back(idx_arr[i]);
        }
        result[i] = arr.at(coords);
    }
    
    return result;
}

// Simple slicing (1D)
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

