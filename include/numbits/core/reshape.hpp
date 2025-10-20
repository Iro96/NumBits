#pragma once
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>  // for std::multiplies
#include "ndarray.hpp"

namespace numbits {

/**
 * @brief Reshape an ndarray to a new shape without copying data.
 *
 * Creates a view of the array with the specified shape without copying underlying data.
 *
 * @tparam T Element type of the ndarray.
 * @param A Input array.
 * @param new_shape Desired shape; each dimension must be greater than zero.
 * @return ndarray<T> An ndarray that shares the same data pointer as the input and has shape `new_shape`.
 *
 * @throws std::invalid_argument if `new_shape` is empty.
 * @throws std::invalid_argument if any dimension in `new_shape` is zero.
 * @throws std::invalid_argument if the total number of elements in `new_shape` does not match the input array's size.
 */
template <typename T>
ndarray<T> reshape(const ndarray<T>& A, const std::vector<size_t>& new_shape) {
    if (new_shape.empty())
        throw std::invalid_argument("reshape: shape cannot be empty");
    if (std::any_of(new_shape.begin(), new_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("reshape: shape dimensions must be > 0");

    size_t old_size = A.size();
    size_t new_size = 1;
    for (size_t d : new_shape) {
        if (d == 0) {
            throw std::invalid_argument("reshape: shape dimensions must be > 0");
        }
        if (d > 0 && new_size > std::numeric_limits<size_t>::max() / d) {
            throw std::invalid_argument("reshape: size overflow in shape product");
        }
        new_size *= d;
    }
    if (old_size != new_size)
        throw std::invalid_argument("reshape: total size must remain the same");

    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Overload of reshape() accepting an initializer list as the new shape.
 * @see reshape(const ndarray<T>&, const std::vector<size_t>&)
 */
template <typename T>
ndarray<T> reshape(const ndarray<T>& A, const std::initializer_list<size_t>& new_shape) {
    return reshape(A, std::vector<size_t>(new_shape));
}

/**
 * @brief Insert a new axis of size 1 at the specified position.
 *
 * Inserts a dimension of size 1 at index `axis` (0-based) and returns a view
 * over the same underlying data with the updated shape.
 *
 * @tparam T Element type.
 * @param A Input array.
 * @param axis Index at which to insert the new axis; allowed values are
 *             0..rank (inclusive), where `rank` is the number of existing axes.
 * @return ndarray<T> Array view with the new axis of length 1 inserted.
 *
 * @throws std::invalid_argument if `axis` is greater than the current rank.
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
 *
 * Removes singleton dimensions from an array view. If `axis` is non-negative,
 * removes the specified axis only; otherwise removes all axes with size 1.
 *
 * @tparam T Element type.
 * @param A Input array.
 * @param axis Index of the axis to remove when >= 0. Negative value (default -1) means remove all size-1 axes.
 * @return ndarray<T> New view with singleton dimensions removed; if all dimensions are removed the result has shape {1}.
 *
 * @throws std::invalid_argument If `axis` is out of bounds or the specified axis does not have size 1.
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
        if (new_shape.empty()) new_shape.push_back(1);
    } else {
        new_shape.erase(std::remove(new_shape.begin(), new_shape.end(), 1), new_shape.end());
        if (new_shape.empty()) new_shape.push_back(1);
    }
    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Transposes a 2D array by swapping its rows and columns.
 *
 * @tparam T Element type of the array.
 * @param A Input array which must be two-dimensional.
 * @return ndarray<T> A new array with shape {cols, rows} where element (j,i) is copied from A(i,j).
 *
 * @throws std::invalid_argument if A is not 2D.
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
 * @brief Broadcasts an array to a larger target shape by expanding size-1 dimensions.
 *
 * Creates a new ndarray with the specified target_shape where dimensions of size 1
 * in the source are replicated to match the corresponding target dimensions; data is
 * copied into the new array according to standard broadcasting rules.
 *
 * @tparam T Element type.
 * @param A Source array to broadcast.
 * @param target_shape Desired shape to broadcast to; must have length >= A.shape().size().
 * @return ndarray<T> Array with shape equal to `target_shape` containing broadcasted values from `A`.
 *
 * @throws std::invalid_argument if `target_shape` is empty.
 * @throws std::invalid_argument if any dimension in `target_shape` is zero.
 * @throws std::invalid_argument if `target_shape` has fewer dimensions than `A`.
 * @throws std::invalid_argument if any non-singleton dimension of `A` is incompatible with the corresponding target dimension.
 */
template <typename T>
ndarray<T> broadcast_to(const ndarray<T>& A, const std::vector<size_t>& target_shape) {
    if (target_shape.empty())
        throw std::invalid_argument("broadcast_to: target shape cannot be empty");
    if (std::any_of(target_shape.begin(), target_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("broadcast_to: shape dimensions must be > 0");

    const auto& orig_shape = A.shape();
    std::vector<size_t> new_shape = target_shape;

    if (orig_shape.size() > new_shape.size())
        throw std::invalid_argument("broadcast_to: target shape must have >= number of dimensions");

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
    if (ndim_src >= 2) {
        for (int i = int(ndim_src) - 2; i >= 0; --i)
            src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

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

/**
 * @brief Overload of broadcast_to() accepting an initializer list as the target shape.
 * @see broadcast_to(const ndarray<T>&, const std::vector<size_t>&)
 */
template <typename T>
ndarray<T> broadcast_to(const ndarray<T>& A, const std::initializer_list<size_t>& target_shape) {
    return broadcast_to(A, std::vector<size_t>(target_shape));
}

/**
 * @brief Extracts a 2D subarray specified by row and column ranges.
 *
 * The row range is [row_start, row_end) and the column range is [col_start, col_end).
 * The returned array contains a copy of the selected region with shape {row_end - row_start, col_end - col_start}.
 *
 * @tparam T Element type.
 * @param A Input 2D array.
 * @param row_start Inclusive start index for rows.
 * @param row_end Exclusive end index for rows.
 * @param col_start Inclusive start index for columns.
 * @param col_end Exclusive end index for columns.
 * @return ndarray<T> A new 2D array containing the sliced region.
 *
 * @throws std::invalid_argument if the input is not 2D or if any start index is greater than its corresponding end index.
 * @throws std::out_of_range if any provided index is outside the bounds of the input array.
 */
template <typename T>
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
