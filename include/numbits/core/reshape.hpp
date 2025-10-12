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
/**
 * Reshape an ndarray to a new shape without copying its underlying data.
 *
 * @param A Source array whose data buffer will be shared by the result.
 * @param new_shape Desired shape; each dimension must be greater than 0 and the product of dimensions must equal the number of elements in `A`.
 * @return ndarray<T> A new array view with shape `new_shape` that shares `A`'s underlying data pointer.
 *
 * @throws std::invalid_argument If `new_shape` is empty, any dimension is 0, or the total number of elements differs from `A`.
 */
ndarray<T> reshape(const ndarray<T>& A, const std::vector<size_t>& new_shape) {
    if (new_shape.empty())
        throw std::invalid_argument("reshape: shape cannot be empty");
    if (std::any_of(new_shape.begin(), new_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("reshape: shape dimensions must be > 0");
    size_t old_size = A.size();
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if (old_size != new_size)
        throw std::invalid_argument("reshape: total size must remain the same");

    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Overload: Accept initializer_list as new shape
 */
template <typename T>
/**
 * @brief Create a view of `A` with the specified shape, sharing the same underlying data.
 *
 * The new shape must be non-empty, all dimensions must be greater than zero, and the
 * product of the dimensions must equal the number of elements in `A`.
 *
 * @param A Source array to reshape.
 * @param new_shape Desired shape as an initializer list.
 * @return ndarray<T> View of `A` with shape `new_shape` that shares `A`'s data.
 * @throws std::invalid_argument If `new_shape` is empty, contains zero dimensions,
 *         or its element count does not match `A`'s total elements.
 */
ndarray<T> reshape(const ndarray<T>& A, const std::initializer_list<size_t>& new_shape) {
    return reshape(A, std::vector<size_t>(new_shape));
}

/**
 * @brief Insert a new axis of size 1 at the specified position.
 */
template <typename T>
/**
 * @brief Insert a new axis of length 1 into the array's shape at the specified position.
 *
 * @param axis Index at which to insert the new axis. Valid values are 0 through the array's current rank (inclusive).
 * @return ndarray<T> A view of the original array with the new axis inserted; shares the same underlying data.
 * @throws std::invalid_argument if `axis` is greater than the array's current rank.
 */
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
/**
 * @brief Remove singleton dimensions from an ndarray, optionally at a specific axis.
 *
 * If `axis` is non-negative, removes the dimension at that index only and requires that
 * the targeted dimension has size 1. If `axis` is negative (default -1), removes all
 * dimensions with size 1.
 *
 * @tparam T Element type of the ndarray.
 * @param axis Index of the axis to remove, or -1 to remove all size-1 axes. Axis is
 *        interpreted as a zero-based index and must be in range [0, ndim-1] when non-negative.
 * @return ndarray<T> New view with the specified singleton dimensions removed that shares
 *         the original array's underlying data pointer.
 * @throws std::invalid_argument if `axis` is out of bounds.
 * @throws std::invalid_argument if `axis` is non-negative and the selected axis has size != 1.
 */
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
/**
 * @brief Transpose a 2D array by swapping its rows and columns.
 *
 * @param A Input array; must have exactly 2 dimensions.
 * @return ndarray<T> A new array with shape {original_cols, original_rows} containing the transposed elements.
 * @throws std::invalid_argument if `A` does not have exactly 2 dimensions.
 */
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
/**
 * @brief Broadcasts an array to a larger shape by expanding singleton dimensions.
 *
 * Creates and returns a new array whose shape equals `target_shape` where each
 * dimension of `A` is either equal to the corresponding (right-aligned) target
 * dimension or equal to 1 and therefore replicated across that axis.
 *
 * @tparam T Element type of the array.
 * @param A Source array to broadcast.
 * @param target_shape Desired shape to broadcast `A` to; must have at least as
 * many dimensions as `A`.
 * @return ndarray<T> A new array with shape `target_shape` containing values
 * copied from `A` according to broadcasting rules.
 * @throws std::invalid_argument If `target_shape` has fewer dimensions than
 * `A`, or if any non-singleton dimension of `A` is incompatible with the
 * corresponding target dimension.
 */
ndarray<T> broadcast_to(const ndarray<T>& A, const std::vector<size_t>& target_shape) {
    const auto& orig_shape = A.shape();
    std::vector<size_t> new_shape = target_shape;

    if (orig_shape.size() > new_shape.size())
        throw std::invalid_argument("broadcast_to: target shape must have >= dimensions");

    size_t align_offset = new_shape.size() - orig_shape.size();
    for (size_t i = 0; i < orig_shape.size(); ++i) {
        if (orig_shape[i] != 1 && orig_shape[i] != new_shape[i + align_offset])
            throw std::invalid_argument("broadcast_to: incompatible shapes");
    }

    ndarray<T> B(new_shape);
    const auto& src = A.data();
    auto& dst = B.data();

    // Prepare aligned shapes
    std::vector<size_t> src_shape = A.shape();
    std::vector<size_t> dst_shape = B.shape();
    size_t ndim_dst = dst_shape.size();
    size_t ndim_src = src_shape.size();
    size_t dst_src_offset = ndim_dst - ndim_src;

    // Compute strides for source in row-major
    std::vector<size_t> src_strides(ndim_src, 1);
    for (int i = int(ndim_src) - 2; i >= 0; --i)
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];

    for (size_t idx = 0; idx < B.size(); ++idx) {
        size_t rem = idx;
        size_t src_flat = 0;
        for (size_t k = 0; k < ndim_dst; ++k) {
            size_t dim = dst_shape[ndim_dst - 1 - k];
            size_t coord = rem % dim;
            rem /= dim;
            if (ndim_dst - 1 - k >= dst_src_offset) {
                size_t s_axis = (ndim_dst - 1 - k) - dst_src_offset;
                size_t s_dim = src_shape[s_axis];
                size_t s_coord = (s_dim == 1) ? 0 : coord;
                src_flat += s_coord * src_strides[s_axis];
            }
        }
        dst[idx] = src[src_flat];
    }

    return B;
}

/** Overload for initializer_list */
template <typename T>
/**
 * @brief Broadcasts an array to the specified target shape given as an initializer list.
 *
 * The returned array shares values from the original array broadcast across any
 * dimensions of size 1 or matching dimensions in the target shape.
 *
 * @param target_shape Desired shape to broadcast to.
 * @return ndarray<T> Array with shape equal to `target_shape` containing the broadcasted values.
 * @throws std::invalid_argument if `target_shape` is not compatible with the source array's shape.
 */
ndarray<T> broadcast_to(const ndarray<T>& A, const std::initializer_list<size_t>& target_shape) {
    return broadcast_to(A, std::vector<size_t>(target_shape));
}

/**
 * @brief Slice a 2D array by row/column ranges (start inclusive, end exclusive).
 */
template <typename T>
/**
 * @brief Extracts a 2D subarray from the given array using half-open row and column ranges.
 *
 * The slice includes rows in [row_start, row_end) and columns in [col_start, col_end).
 *
 * @param row_start Start row index (inclusive).
 * @param row_end End row index (exclusive).
 * @param col_start Start column index (inclusive).
 * @param col_end End column index (exclusive).
 * @return ndarray<T> A new 2D array containing the requested slice.
 *
 * @throws std::invalid_argument if the input is not 2D or if any start index is greater than its corresponding end index.
 * @throws std::out_of_range if any index is outside the bounds of the input array.
 */
ndarray<T> slice(const ndarray<T>& A, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    const auto& shape = A.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("slice: only 2D arrays supported");
    if (row_start > row_end || col_start > col_end)
        throw std::invalid_argument("slice: start must be <= end");
    if (row_end > shape[0] || col_end > shape[1] || row_start >= shape[0] || col_start >= shape[1])
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