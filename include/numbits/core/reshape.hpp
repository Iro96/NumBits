#pragma once
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits>
#include "ndarray.hpp"
#include "shape.hpp"

namespace numbits {

/**
 * @brief Return a new array with the same data but a different shape.
 *
 * Creates an ndarray that **shares** the underlying data of A, but is interpreted
 * with the supplied `new_shape`. The total element count must match.
 *
 * @tparam T Element type of the array.
 * @param A The input ndarray.
 * @param new_shape A vector of sizes specifying the target shape. Must be non‐empty,
 *                  each dimension > 0, and product of dims must equal A.size().
 * @return An ndarray<T> with shape equal to new_shape and shared data pointer from A.
 * @throws std::invalid_argument if new_shape is empty, any dimension is zero,
 *         or the total size does not match A.size().
 */
template <typename T>
inline ndarray<T> reshape(const ndarray<T>& A, const std::vector<size_t>& new_shape) {
    if (new_shape.empty())
        throw std::invalid_argument("reshape: shape cannot be empty");
    if (std::any_of(new_shape.begin(), new_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("reshape: shape dimensions must be > 0");
    if (A.size() != numbits::total_size(new_shape))
        throw std::invalid_argument("reshape: total size must remain the same");
    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Overload of reshape allowing initializer‐list for new shape.
 *
 * @tparam T Element type of the array.
 * @param A The input ndarray.
 * @param new_shape An initializer_list of sizes specifying the target shape.
 * @return An ndarray<T> with shape equal to new_shape and shared data pointer from A.
 * @throws std::invalid_argument under same conditions as the vector‐based overload.
 */
template <typename T>
inline ndarray<T> reshape(const ndarray<T>& A, const std::initializer_list<size_t>& new_shape) {
    return reshape(A, std::vector<size_t>(new_shape));
}

/**
 * @brief Insert a new axis of size 1 into the array’s shape at the given position.
 *
 * Creates a “view” that shares A’s data but has one additional dimension of size 1
 * at index `axis`.
 *
 * @tparam T Element type of the array.
 * @param A The input ndarray.
 * @param axis Position at which to insert the new dimension. Must be ≤ A.rank().
 * @return An ndarray<T> sharing data with A but having one extra size‑1 dimension.
 * @throws std::invalid_argument if axis > A.rank().
 */
template <typename T>
inline ndarray<T> expand_dims(const ndarray<T>& A, size_t axis) {
    std::vector<size_t> new_shape = A.shape();
    if (axis > new_shape.size())
        throw std::invalid_argument("expand_dims: axis out of bounds");
    new_shape.insert(new_shape.begin() + axis, 1);
    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Remove size‑1 dimensions (singleton axes) from the shape.
 *
 * If `axis >= 0`, then only the specified axis is removed (if it is size 1). If `axis == –1`,
 * then **all** size‑1 axes are removed. The result must remain at least rank 1.
 *
 * @tparam T Element type of the array.
 * @param A The input ndarray.
 * @param axis The axis index to remove (non‐negative) or –1 to remove all singleton axes.
 * @return An ndarray<T> sharing data with A but with reduced dimensionality.
 * @throws std::invalid_argument if axis out of bounds, or if axis dimension isn’t 1,
 *         or if removal result would produce a 0‐D array.
 */
template <typename T>
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
            throw std::invalid_argument("squeeze: removing this axis would produce 0‑D; not supported");
    } else {
        new_shape.erase(std::remove(new_shape.begin(), new_shape.end(), 1), new_shape.end());
        if (new_shape.empty())
            throw std::invalid_argument("squeeze: removing all singleton axes would produce 0‑D; not supported");
    }
    return ndarray<T>(new_shape, A.data_ptr());
}

/**
 * @brief Transpose a 2‑dimensional array (swap its two axes).
 *
 * Only supports 2D arrays (shape.size() == 2). Produces a new array,
 * not a view, with its own data buffer.
 *
 * @tparam T Element type of the array.
 * @param A Input ndarray of shape {rows, cols}.
 * @return New ndarray<T> of shape {cols, rows} with element (i,j) mapped to (j,i).
 * @throws std::invalid_argument if A is not 2‑D.
 */
template <typename T>
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

/**
 * @brief Broadcast array A to a new shape, if dimensions are compatible.
 *
 * Mimics NumPy’s broadcasting rules (a bit simplified). The target shape must
 * have at least as many dimensions as A, and for each dimension either the
 * original size is 1 or matches the target size. The returned array has its
 * own data buffer and contains repeated values accordingly.
 *
 * @tparam T Element type of the array.
 * @param A Input ndarray.
 * @param target_shape Desired output shape (vector form). Must be non‑empty, all dims > 0,
 *                     and compatible with A.shape().
 * @return New ndarray<T> of shape target_shape with broadcasted data.
 * @throws std::invalid_argument if target_shape is empty, any dimension is zero,
 *         target rank is less than A.rank, or shapes are incompatible for broadcasting.
 */
template <typename T>
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

/**
 * @brief Broadcast array A to a new shape, using initializer‐list for target shape.
 *
 * @tparam T Element type of the array.
 * @param A Input ndarray.
 * @param target_shape An initializer_list of sizes specifying the target shape.
 * @return New ndarray<T> of shape target_shape with broadcasted data.
 * @throws std::invalid_argument under the same conditions as the vector‐based overload.
 */
template <typename T>
inline ndarray<T> broadcast_to(const ndarray<T>& A, const std::initializer_list<size_t>& target_shape) {
    return broadcast_to(A, std::vector<size_t>(target_shape));
}

/**
 * @brief Slice a 2‑D array by selecting a contiguous sub‐block.
 *
 * The input must be a 2D array (shape size == 2). Returns a new array (not a view)
 * of shape (row_end‑row_start, col_end‑col_start) containing the block from A
 * starting at (row_start, col_start) up to (row_end‑1, col_end‑1).
 *
 * @tparam T Element type of the array.
 * @param A Input 2‑D ndarray.
 * @param row_start Zero‐based starting row index (inclusive).
 * @param row_end Zero‐based ending row index (exclusive).
 * @param col_start Zero‐based starting column index (inclusive).
 * @param col_end Zero‐based ending column index (exclusive).
 * @return New ndarray<T> of shape {row_end‑row_start, col_end‑col_start} containing the slice.
 * @throws std::invalid_argument if A is not 2‑D, or if start > end.
 * @throws std::out_of_range if row_end > A.shape()[0] or col_end > A.shape()[1].
 */
template <typename T>
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
