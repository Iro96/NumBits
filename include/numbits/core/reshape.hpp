#pragma once
#include "ndarray.hpp"
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace numbits {

/**
 * @brief Reshape an ndarray to a new shape without copying data.
 * @param A Input array
 * @param new_shape Desired shape (total size must match)
 * @return ndarray<T> sharing the same underlying data
 */
template <typename T>
ndarray<T> reshape(const ndarray<T>& A, const std::initializer_list<size_t>& new_shape) {
    size_t old_size = A.size();
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if (old_size != new_size)
        throw std::invalid_argument("reshape: total size must remain the same");

    return ndarray<T>(std::vector<size_t>(new_shape), A.data_ptr());
}

/**
 * @brief Insert a new axis of size 1 at the specified position.
 */
template <typename T>
ndarray<T> expand_dims(const ndarray<T>& A, size_t axis) {
    std::vector<size_t> new_shape = A.shape();
    if (axis > new_shape.size())
        throw std::invalid_argument("expand_dims: axis out of bounds");
    new_shape.insert(new_shape.begin() + axis, 1);
    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Remove axes of size 1. If axis >= 0, remove only that axis.
 */
template <typename T>
ndarray<T> squeeze(const ndarray<T>& A, int axis = -1) {
    std::vector<size_t> new_shape = A.shape();
    if (axis >= 0) {
        if (axis >= static_cast<int>(new_shape.size()))
            throw std::invalid_argument("squeeze: axis out of bounds");
        if (new_shape[axis] != 1)
            throw std::invalid_argument("squeeze: cannot squeeze axis with size != 1");
        new_shape.erase(new_shape.begin() + axis);
    } else {
        new_shape.erase(std::remove(new_shape.begin(), new_shape.end(), 1), new_shape.end());
    }
    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Transpose a 2D array (swap rows and columns).
 */
template <typename T>
ndarray<T> transpose(const ndarray<T>& A) {
    const auto& shape = A.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("transpose: only 2D arrays supported");

    ndarray<T> B({shape[1], shape[0]});
    for (size_t i = 0; i < shape[0]; ++i)
        for (size_t j = 0; j < shape[1]; ++j)
            B(j, i) = A(i, j);

    return B;
}

/**
 * @brief Broadcast array to a new shape (size-1 axes can expand).
 */
template <typename T>
ndarray<T> broadcast_to(const ndarray<T>& A, const std::initializer_list<size_t>& target_shape) {
    const auto& orig_shape = A.shape();
    std::vector<size_t> new_shape(target_shape);

    if (orig_shape.size() > new_shape.size())
        throw std::invalid_argument("broadcast_to: target shape must have >= dimensions");

    size_t offset = new_shape.size() - orig_shape.size();
    for (size_t i = 0; i < orig_shape.size(); ++i) {
        if (orig_shape[i] != 1 && orig_shape[i] != new_shape[i + offset])
            throw std::invalid_argument("broadcast_to: incompatible shapes");
    }

    ndarray<T> B({new_shape.begin(), new_shape.end()});
    const auto& src = A.data();
    auto& dst = B.data();
    for (size_t idx = 0; idx < B.size(); ++idx)
        dst[idx] = src[idx % A.size()];

    return B;
}

/**
 * @brief Slice a 2D array by row/column ranges (start inclusive, end exclusive).
 */
template <typename T>
ndarray<T> slice(const ndarray<T>& A, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    const auto& shape = A.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("slice: only 2D arrays supported");
    if (row_end > shape[0] || col_end > shape[1])
        throw std::out_of_range("slice: indices out of bounds");

    size_t rows = row_end - row_start;
    size_t cols = col_end - col_start;
    ndarray<T> B({rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            B(i, j) = A(i + row_start, j + col_start);

    return B;
}

} // namespace numbits
