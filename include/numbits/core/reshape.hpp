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
 *
 * @note The returned ndarray is a non-owning view sharing the same data pointer.
 *       Ensure the source array remains alive for the lifetime of the view.
 *
 * @code
 * ndarray<double> A({2, 3});
 * auto B = numbits::reshape(A, {3, 2});
 * // A.shape() == {2, 3}
 * // B.shape() == {3, 2}
 * @endcode
 */
template <typename T>
ndarray<T> reshape(const ndarray<T>& A, const std::vector<size_t>& new_shape) {
    if (std::any_of(new_shape.begin(), new_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("reshape: shape dimensions must be > 0");

    // Validate total size consistency
    if (A.size() != numbits::total_size(new_shape))
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
 *
 * @note The returned ndarray is a view sharing the same data pointer.
 *
 * @code
 * ndarray<int> A({2, 3});
 * auto B = numbits::expand_dims(A, 0);
 * // A.shape() == {2, 3}
 * // B.shape() == {1, 2, 3}
 * @endcode
 */
template <typename T>
constexpr ndarray<T> expand_dims(const ndarray<T>& A, size_t axis) {
    auto shape_vec = A.shape();
    if (axis > shape_vec.size())
        throw std::invalid_argument("expand_dims: axis out of bounds");
    shape_vec.insert(shape_vec.begin() + axis, 1);
    return ndarray<T>(shape_vec, A.data_ptr());
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
 *
 * @note The returned ndarray is a view sharing the same data pointer.
 *
 * @code
 * ndarray<int> A({1, 3, 1, 4});
 * auto B = numbits::squeeze(A);     // shape {3, 4}
 * auto C = numbits::squeeze(A, 2);  // remove axis 2 only
 * @endcode
 */
template <typename T>
constexpr ndarray<T> squeeze(const ndarray<T>& A, int axis = -1) {
    auto new_shape = A.shape();

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

/**
 * @brief Transposes a 2D array by swapping its rows and columns.
 *
 * @tparam T Element type of the array.
 * @param A Input array which must be two-dimensional.
 * @return ndarray<T> A new array with shape {cols, rows} where element (j,i) is copied from A(i,j).
 *
 * @throws std::invalid_argument if A is not 2D.
 *
 * @code
 * ndarray<double> A({2, 3});
 * auto B = numbits::transpose(A);
 * // A(i, j) == B(j, i)
 * @endcode
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
 * Creates a new ndarray with the specified `target_shape` where dimensions of size 1
 * in the source are replicated to match the corresponding target dimensions.
 *
 * @tparam T Element type.
 * @param A Source array to broadcast.
 * @param target_shape Desired shape to broadcast to; must have length >= A.shape().size().
 * @return ndarray<T> Array with shape equal to `target_shape` containing broadcasted values from `A`.
 *
 * @throws std::invalid_argument if `target_shape` is empty or has zero dimensions.
 * @throws std::invalid_argument if `target_shape` has fewer dimensions than `A`.
 * @throws std::invalid_argument if any non-singleton dimension of `A` is incompatible with the corresponding target dimension.
 *
 * @code
 * ndarray<int> A({1, 3});
 * auto B = numbits::broadcast_to(A, {4, 3});
 * // B(i, j) == A(0, j)
 * @endcode
 */
template <typename T>
ndarray<T> broadcast_to(const ndarray<T>& A, const std::vector<size_t>& target_shape) {
    if (target_shape.empty())
        throw std::invalid_argument("broadcast_to: target shape cannot be empty");
    if (std::any_of(target_shape.begin(), target_shape.end(), [](size_t d){ return d == 0; }))
        throw std::invalid_argument("broadcast_to: shape dimensions must be > 0");

    // Check for overflow in target shape product
    size_t target_size = 1;
    for (size_t dim : target_shape) {
        if (dim > std::numeric_limits<size_t>::max() / target_size)
            throw std::overflow_error("broadcast_to: target shape product overflow");
        target_size *= dim;
    }

    const auto& orig_shape = A.shape();
    if (orig_shape.size() > target_shape.size())
        throw std::invalid_argument("broadcast_to: target shape must have >= number of dimensions");

    size_t align_offset = target_shape.size() - orig_shape.size();
    for (size_t i = 0; i < orig_shape.size(); ++i) {
        if (orig_shape[i] != 1 && orig_shape[i] != target_shape[i + align_offset])
            throw std::invalid_argument("broadcast_to: incompatible shapes");
    }

    // Precompute repetition patterns for each axis
    std::vector<size_t> repetitions(target_shape.size());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (i < align_offset) {
            repetitions[i] = target_shape[i];
        } else {
            size_t src_dim = orig_shape[i - align_offset];
            repetitions[i] = (src_dim == 1) ? target_shape[i] : 1;
        }
    }

    // Compute strides for target shape with overflow check
    std::vector<size_t> target_strides_vec(target_shape.size());
    if (!target_shape.empty()) {
        target_strides_vec.back() = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(target_shape.size()) - 2; i >= 0; --i) {
            if (target_shape[i + 1] > std::numeric_limits<size_t>::max() / target_strides_vec[i + 1])
                throw std::overflow_error("broadcast_to: stride computation overflow");
            target_strides_vec[i] = target_strides_vec[i + 1] * target_shape[i + 1];
        }
    }
    auto target_strides = std::make_shared<std::vector<size_t>>(target_strides_vec);

    ndarray<T> B(target_shape, target_strides, A.data_ptr());
    const auto& src = A.data();
    auto& dst = const_cast<std::vector<T>&>(B.data());

    // Use row-major contiguous loops for innermost dimensions
    size_t total_size = B.size();
    size_t inner_dim = target_shape.back();
    size_t outer_size = total_size / inner_dim;

    for (size_t outer = 0; outer < outer_size; ++outer) {
        size_t src_idx = 0;
        size_t temp = outer;
        for (size_t k = 0; k < target_shape.size() - 1; ++k) {
            size_t coord = temp % target_shape[k];
            temp /= target_shape[k];
            if (k >= align_offset) {
                size_t s_axis = k - align_offset;
                size_t s_dim = orig_shape[s_axis];
                size_t s_coord = (s_dim == 1) ? 0 : coord;
                src_idx += s_coord * (*A.strides())[s_axis];
            }
        }
        for (size_t inner = 0; inner < inner_dim; ++inner) {
            dst[outer * inner_dim + inner] = src[src_idx];
        }
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
 *
 * @code
 * ndarray<int> A({4, 5});
 * auto sub = numbits::slice(A, 1, 3, 0, 2);
 * // sub.shape() == {2, 2}
 * @endcode
 */
template <typename T>
ndarray<T> slice(const ndarray<T>& A, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    const auto& shape = A.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("slice: only 2D arrays supported");
    if (row_start > row_end || col_start > col_end)
        throw std::invalid_argument("slice: start must be <= end");
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
