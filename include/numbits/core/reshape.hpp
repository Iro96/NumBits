#pragma once
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits>
#include "ndarray.hpp"
#include "shape.hpp"

namespace numbits {

template <typename T>
/**
 * @brief Create a view of an array with a different shape.
 *
 * Produces a new ndarray that shares the same underlying data as `A` but presents it with `new_shape`.
 *
 * @param A Source array whose data buffer will be reused.
 * @param new_shape Desired shape; must be non-empty and contain only dimensions greater than zero.
 * @return ndarray<T> Array view with shape `new_shape` backed by `A`'s data.
 *
 * @throws std::invalid_argument if `new_shape` is empty, contains a zero dimension, or its total size does not match `A.size()`.
 */
inline ndarray<T> reshape(const ndarray<T>& A, const std::vector<size_t>& new_shape) {
    if (new_shape.empty())
        throw std::invalid_argument("reshape: shape cannot be empty");
    if (std::any_of(new_shape.begin(), new_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("reshape: shape dimensions must be > 0");
    if (A.size() != numbits::total_size(new_shape))
        throw std::invalid_argument("reshape: total size must remain the same");
    return ndarray<T>(new_shape, A.data_ptr());
}

template <typename T>
/**
 * @brief Create a view of an array with a different shape specified by an initializer list.
 *
 * Creates a new ndarray that shares A's underlying data but has the shape given by `new_shape`.
 * The total number of elements implied by `new_shape` must equal A.size(); each dimension must be greater than zero.
 *
 * @param A Source array whose data will be reused.
 * @param new_shape Desired shape for the returned array.
 * @return ndarray<T> New array view sharing A's data with the specified shape.
 */
inline ndarray<T> reshape(const ndarray<T>& A, const std::initializer_list<size_t>& new_shape) {
    return reshape(A, std::vector<size_t>(new_shape));
}

template <typename T>
/**
 * Create a view of A with a new singleton dimension inserted at the specified axis.
 *
 * @tparam T Element type of the array.
 * @param A Input array whose shape will be expanded.
 * @param axis Position at which to insert a new dimension of size 1; valid range is [0, rank],
 *             where rank is the number of dimensions of `A`.
 * @return ndarray<T> New array sharing `A`'s data pointer but with a shape that includes a size-1
 *         dimension at `axis`.
 * @throws std::invalid_argument if `axis` is greater than the number of dimensions of `A`.
 */
inline ndarray<T> expand_dims(const ndarray<T>& A, size_t axis) {
    std::vector<size_t> new_shape = A.shape();
    if (axis > new_shape.size())
        throw std::invalid_argument("expand_dims: axis out of bounds");
    new_shape.insert(new_shape.begin() + axis, 1);
    return ndarray<T>(new_shape, A.data_ptr());
}

template <typename T>
/**
 * @brief Remove singleton dimensions from an array.
 *
 * Removes dimensions of size 1 from A. If `axis` >= 0, removes only the specified axis (must be within range and have size 1); if `axis` < 0, removes all singleton axes. The returned ndarray shares A's data pointer but has the squeezed shape.
 *
 * @tparam T Element type of the ndarray.
 * @param A Input array.
 * @param axis Axis index to remove (0-based). A negative value (default -1) indicates that all singleton axes should be removed.
 * @return ndarray<T> Array with singleton dimensions removed.
 * @throws std::invalid_argument if `axis` is out of bounds, if the specified axis does not have size 1, or if removing the requested axes would produce a 0-D array.
 */
inline ndarray<T> squeeze(const ndarray<T>& A, int axis = -1) {
    std::vector<size_t> new_shape = A.shape();
    if (axis >= 0) {
        const size_t uaxis = static_cast<size_t>(axis);
        if (uaxis >= new_shape.size())
            throw std::invalid_argument("squeeze: axis out of bounds");
        if (new_shape[uaxis] != 1)
            throw std::invalid_argument("squeeze: cannot squeeze axis with size != 1");
        new_shape.erase(new_shape.begin() + static_cast<std::ptrdiff_t>(uaxis));
        if (new_shape.empty())
            throw std::invalid_argument("squeeze: removing this axis would produce 0-D; not supported");
    } else {
        new_shape.erase(std::remove(new_shape.begin(), new_shape.end(), 1), new_shape.end());
        if (new_shape.empty())
            throw std::invalid_argument("squeeze: removing all singleton axes would produce 0-D; not supported");
    }
    return ndarray<T>(new_shape, A.data_ptr());
}

template <typename T>
/**
 * @brief Compute the transpose of a 2D array.
 *
 * @param A Input 2-dimensional ndarray to transpose.
 * @return ndarray<T> New array whose shape is {cols, rows} and whose (i,j) element equals A(j,i).
 *
 * @throws std::invalid_argument if A is not 2-dimensional.
 */
inline ndarray<T> transpose(const ndarray<T>& A) {
    const auto& shape = A.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("transpose: only 2D arrays supported");
    const size_t rows = shape[0], cols = shape[1];
    ndarray<T> B({cols, rows});
    const auto& Ad = A.data();
    auto& Bd = B.data();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            Bd[j * rows + i] = Ad[i * cols + j];
    return B;
}

template <typename T>
/**
 * @brief Broadcasts an array to a specified shape.
 *
 * Creates a new array whose shape is `target_shape` and whose values are the result
 * of broadcasting the elements of `A` across the new shape following trailing-dimension
 * broadcasting rules.
 *
 * @tparam T Element type of the array.
 * @param A Source array to broadcast.
 * @param target_shape Desired shape for the result; must be non-empty and all dimensions > 0.
 * @return ndarray<T> Array with shape equal to `target_shape` containing values from `A`
 *         broadcast to that shape.
 * @throws std::invalid_argument if `target_shape` is empty.
 * @throws std::invalid_argument if any dimension in `target_shape` is zero.
 * @throws std::invalid_argument if `target_shape` has fewer dimensions than `A`.
 * @throws std::invalid_argument if any non-singleton dimension of `A` does not match the
 *         corresponding trailing dimension of `target_shape`.
 */
inline ndarray<T> broadcast_to(const ndarray<T>& A, const std::vector<size_t>& target_shape) {
    if (target_shape.empty())
        throw std::invalid_argument("broadcast_to: target shape cannot be empty");
    if (std::any_of(target_shape.begin(), target_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("broadcast_to: shape dimensions must be > 0");

    const auto& orig_shape = A.shape();
    if (orig_shape.size() > target_shape.size())
        throw std::invalid_argument("broadcast_to: target shape must have >= number of dimensions");

    const size_t align_offset = target_shape.size() - orig_shape.size();
    for (size_t i = 0; i < orig_shape.size(); ++i)
        if (orig_shape[i] != 1 && orig_shape[i] != target_shape[i + align_offset])
            throw std::invalid_argument("broadcast_to: incompatible shapes");

    ndarray<T> B(target_shape);
    const auto* src_ptr = A.data().data();
    auto* dst_ptr = B.data().data();

    std::vector<size_t> src_shape = orig_shape;
    std::vector<size_t> dst_shape = B.shape();
    const size_t ndim_dst = dst_shape.size(), ndim_src = src_shape.size(), dst_src_offset = ndim_dst - ndim_src;

    std::vector<size_t> src_strides(ndim_src, 1);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(ndim_src) - 2; i >= 0; --i)
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];

    for (size_t idx = 0; idx < B.size(); ++idx) {
        size_t rem = idx, src_flat = 0;
        for (size_t k = 0; k < ndim_dst; ++k) {
            const size_t dim = dst_shape[ndim_dst - 1 - k];
            const size_t coord = rem % dim;
            rem /= dim;
            if (ndim_dst - 1 - k >= dst_src_offset) {
                const size_t s_axis = (ndim_dst - 1 - k) - dst_src_offset;
                const size_t s_dim = src_shape[s_axis];
                const size_t s_coord = (s_dim == 1) ? 0 : coord;
                src_flat += s_coord * src_strides[s_axis];
            }
        }
        dst_ptr[idx] = src_ptr[src_flat];
    }
    return B;
}

template <typename T>
/**
 * @brief Broadcasts array A to the specified target shape.
 *
 * @tparam T Element type of the array.
 * @param target_shape Desired shape to broadcast A to; dimensions align with A's trailing dimensions.
 * @return ndarray<T> New array with shape equal to target_shape containing values of A broadcast according to broadcasting compatibility rules.
 */
inline ndarray<T> broadcast_to(const ndarray<T>& A, const std::initializer_list<size_t>& target_shape) {
    return broadcast_to(A, std::vector<size_t>(target_shape));
}

template <typename T>
/**
 * @brief Extracts a 2D subarray from the given array.
 *
 * Creates and returns a new 2D ndarray containing the rows in the half-open range
 * [row_start, row_end) and columns in the half-open range [col_start, col_end) from A.
 *
 * @param row_start Index of the first row to include (inclusive).
 * @param row_end Index one past the last row to include (exclusive).
 * @param col_start Index of the first column to include (inclusive).
 * @param col_end Index one past the last column to include (exclusive).
 * @return ndarray<T> A new ndarray with shape {row_end - row_start, col_end - col_start}
 *
 * @throws std::invalid_argument if A is not 2D or if a start index is greater than its corresponding end index.
 * @throws std::out_of_range if any end index exceeds A's corresponding dimension size.
 */
inline ndarray<T> slice(const ndarray<T>& A, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    const auto& shape = A.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("slice: only 2D arrays supported");
    if (row_start > row_end || col_start > col_end)
        throw std::invalid_argument("slice: start must be <= end");
    if (row_end > shape[0] || col_end > shape[1])
        throw std::out_of_range("slice: indices out of bounds");

    const size_t rows = row_end - row_start;
    const size_t cols = col_end - col_start;
    ndarray<T> B({rows, cols});
    const auto& Ad = A.data();
    auto& Bd = B.data();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            Bd[i * cols + j] = Ad[(i + row_start) * shape[1] + (j + col_start)];
    return B;
}

} // namespace numbits