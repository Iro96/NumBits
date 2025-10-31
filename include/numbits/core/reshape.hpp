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
inline ndarray<T> reshape(const ndarray<T>& A, const std::initializer_list<size_t>& new_shape) {
    return reshape(A, std::vector<size_t>(new_shape));
}

template <typename T>
constexpr inline ndarray<T> expand_dims(const ndarray<T>& A, size_t axis) {
    std::vector<size_t> new_shape = A.shape();
    if (axis > new_shape.size())
        throw std::invalid_argument("expand_dims: axis out of bounds");
    new_shape.insert(new_shape.begin() + axis, 1);
    return ndarray<T>(new_shape, A.data_ptr());
}

template <typename T>
constexpr inline ndarray<T> squeeze(const ndarray<T>& A, int axis = -1) {
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
    for (int i = int(ndim_src) - 2; i >= 0; --i)
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
inline ndarray<T> broadcast_to(const ndarray<T>& A, const std::initializer_list<size_t>& target_shape) {
    return broadcast_to(A, std::vector<size_t>(target_shape));
}

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
