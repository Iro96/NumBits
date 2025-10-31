#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <algorithm>
#include <type_traits>
#include <memory>
#include <tuple>
#include <omp.h>
#include "utils.hpp"

namespace numbits {

/**
 * @brief Simple slice object for array slicing, similar to NumPy's slice.
 */
struct slice {
    std::ptrdiff_t start = 0;
    std::ptrdiff_t stop = -1;
    std::ptrdiff_t step = 1;
};

/**
 * @brief Simple n-dimensional array container with shared underlying data.
 *
 * Provides basic support for multi-dimensional arrays with contiguous storage,
 * N-dimensional indexing, shared data ownership (via std::shared_ptr),
 * and shape/stride metadata similar to NumPy.
 *
 * @tparam T Type of stored elements.
 */
template <typename T>
class ndarray {
public:
    // ---------------- Constructors ----------------

    /**
     * @brief Default-constructs an empty ndarray with no shape and empty data.
     *
     * Creates an ndarray with shared empty storage (`std::shared_ptr<std::vector<T>>`).
     * Accessing elements is invalid until initialized with a shape.
     */
    ndarray() : shape_(), data_(std::make_shared<std::vector<T>>()) {}

    /**
     * @brief Constructs an n-dimensional array with the specified shape and fills it with a value.
     *
     * @param shape List of dimensions; for 0-D arrays, use empty list {}; each dimension > 0 otherwise.
     * @param init  Value used to initialize all elements (defaults to `T()`).
     *
     * @throws std::invalid_argument if any dimension is 0.
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(small_vector<size_t>(shape.begin(), shape.end()), init);
    }

    /**
     * @brief Constructs an n-dimensional array with the given shape and fills it with a value.
     *
     * @param shape Vector of dimension sizes; for 0-D arrays, use empty vector; each dimension > 0 otherwise.
     * @param init  Value used to initialize all elements (defaults to `T()`).
     *
     * @throws std::invalid_argument if any dimension is 0.
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(small_vector<size_t>(shape.begin(), shape.end()), init);
    }

    /**
     * @brief Constructs an ndarray that wraps an existing shared data buffer.
     *
     * @param shape     Dimensions of the array; empty for 0-D scalars.
     * @param data_ptr  Shared pointer to contiguous vector; must match shape size.
     *
     * @throws std::invalid_argument if shape or data pointer is invalid.
     */
    ndarray(const std::vector<size_t>& shape, std::shared_ptr<std::vector<T>> data_ptr)
        : shape_(small_vector<size_t>(shape.begin(), shape.end())), data_(std::move(data_ptr)) {
        if (std::any_of(shape_.begin(), shape_.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        if (!data_)
            throw std::invalid_argument("ndarray: data_ptr cannot be null");
        size_t expected = 1;
        for (size_t dim : shape) {
            if (dim > std::numeric_limits<size_t>::max() / expected)
                throw std::overflow_error("ndarray: shape product overflow");
            expected *= dim;
        }
        if (data_->size() != expected)
            throw std::invalid_argument("ndarray: data size does not match shape");
        compute_strides();
    }

    /**
     * @brief Constructs an ndarray view with shared strides.
     *
     * @param shape     Dimensions of the array; empty for 0-D scalars.
     * @param strides_ptr Shared pointer to strides vector.
     * @param data_ptr  Shared pointer to contiguous vector; must match shape size.
     *
     * @throws std::invalid_argument if shape or data pointer is invalid.
     */
    ndarray(const std::vector<size_t>& shape, std::shared_ptr<std::vector<size_t>> strides_ptr, std::shared_ptr<std::vector<T>> data_ptr)
        : shape_(small_vector<size_t>(shape.begin(), shape.end())), strides_(std::make_shared<small_vector<size_t>>(strides_ptr->begin(), strides_ptr->end())), data_(std::move(data_ptr)) {
        if (std::any_of(shape_.begin(), shape_.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        if (!data_)
            throw std::invalid_argument("ndarray: data_ptr cannot be null");
        size_t expected = 1;
        for (size_t dim : shape) {
            if (dim > std::numeric_limits<size_t>::max() / expected)
                throw std::overflow_error("ndarray: shape product overflow");
            expected *= dim;
        }
        if (data_->size() != expected)
            throw std::invalid_argument("ndarray: data size does not match shape");
        if (!strides_ptr)
            throw std::invalid_argument("ndarray: strides_ptr cannot be null");
        if (strides_ptr->size() != shape_.size())
            throw std::invalid_argument("ndarray: strides size does not match shape");
    }

    /**
     * @brief Constructs an ndarray from a shape and initializer list of values.
     *
     * The number of values must exactly match the product of the shape dimensions.
     *
     * @param shape  List of dimension sizes; empty for 0-D scalars.
     * @param values List of values to initialize the data.
     *
     * @throws std::invalid_argument if sizes do not match.
     */
    ndarray(std::initializer_list<size_t> shape, std::initializer_list<T> values) {
        small_vector<size_t> shape_vec(shape.begin(), shape.end());
        shape_ = shape_vec;
        if (std::any_of(shape_.begin(), shape_.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");

        size_t total = 1;
        for (size_t dim : shape_) {
            if (dim > std::numeric_limits<size_t>::max() / total)
                throw std::overflow_error("ndarray: shape product overflow");
            total *= dim;
        }
        if (values.size() != total)
            throw std::invalid_argument("ndarray: number of values does not match shape size");

        data_ = std::make_shared<std::vector<T>>();
        data_->reserve(total);
        data_->assign(values.begin(), values.end());
        compute_strides();
    }

    // ---------------- Accessors ----------------

    /**
     * @brief Returns the array shape.
     */
    const std::vector<size_t> shape() const noexcept { return std::vector<size_t>(shape_.begin(), shape_.end()); }

    /**
     * @brief Returns the number of dimensions (rank).
     */
    size_t rank() const noexcept { return shape_.size(); }

    /**
     * @brief Returns the number of dimensions (alias for rank).
     */
    size_t ndim() const noexcept { return rank(); }

    /**
     * @brief Checks if the array is C-contiguous (row-major).
     */
    bool is_contiguous() const {
        if (shape_.empty()) return true;
        small_vector<size_t> expected_strides(shape_.size());
        expected_strides.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            expected_strides[i] = expected_strides[i + 1] * shape_[i + 1];
        }
        return std::equal(strides_->begin(), strides_->end(), expected_strides.begin());
    }

    /**
     * @brief Returns the total number of elements.
     */
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /**
     * @brief 1D element access by flat index.
     * @throws std::out_of_range if index is out of bounds.
     */
    T& operator[](size_t i) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        if (i >= data_->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*data_)[i];
    }

    /**
     * @brief Const 1D element access by flat index.
     * @throws std::out_of_range if index is out of bounds.
     */
    const T& operator[](size_t i) const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        if (i >= data_->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*data_)[i];
    }

    /**
     * @brief Multi-dimensional element access using variadic indices.
     *
     * Example:
     * ```cpp
     * ndarray<int> a({2,3}, 0);
     * a(1,2) = 42;
     * ```
     *
     * @throws std::invalid_argument if number of indices doesn't match rank.
     * @throws std::out_of_range if any index is out of bounds.
     */
    template <typename... Idxs>
    T& operator()(Idxs... indices) {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...), "All indices must be size_t");
        if (sizeof...(indices) != shape_.size()) {
            throw std::invalid_argument("ndarray: number of indices does not match rank");
        }

        size_t flat_idx = 0;
        size_t i = 0;
        auto calculate_offset = [&](size_t idx) {
            if (idx >= shape_[i]) {
                throw std::out_of_range("ndarray: index out of bounds");
            }
            flat_idx += idx * (*strides_)[i];
            i++;
        };

        (calculate_offset(static_cast<size_t>(indices)), ...);

        return (*data_)[flat_idx];
    }

    /**
     * @brief Access for 0-D scalars (empty shape).
     *
     * @return Reference to the single element.
     */
    T& operator()() {
        if (!shape_.empty()) {
            throw std::invalid_argument("ndarray: operator()() only valid for 0-D arrays");
        }
        return (*data_)[0];
    }

    /**
     * @brief Const multi-dimensional access.
     */
    template <typename... Idxs>
    const T& operator()(Idxs... indices) const {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...), "All indices must be size_t");
        if (sizeof...(indices) != shape_.size()) {
            throw std::invalid_argument("ndarray: number of indices does not match rank");
        }

        size_t flat_idx = 0;
        size_t i = 0;
        auto calculate_offset = [&](size_t idx) {
            if (idx >= shape_[i])
                throw std::out_of_range("ndarray: index out of bounds");
            flat_idx += idx * (*strides_)[i];
            ++i;
        };

        (calculate_offset(static_cast<size_t>(indices)), ...);
        return (*data_)[flat_idx];
    }

    /**
     * @brief Const access for 0-D scalars (empty shape).
     *
     * @return Const reference to the single element.
     */
    const T& operator()() const {
        if (!shape_.empty()) {
            throw std::invalid_argument("ndarray: operator()() const only valid for 0-D arrays");
        }
        return (*data_)[0];
    }

    /**
     * @brief Returns const reference to internal data vector.
     * @note For safety, only const access is provided to prevent external resizing.
     */
    const std::vector<T>& data() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /**
     * @brief Returns the shared_ptr to the internal data vector.
     */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /**
     * @brief Returns const reference to strides vector.
     */
    const std::shared_ptr<std::vector<size_t>> strides() const noexcept {
        if (!strides_) return nullptr;
        return std::make_shared<std::vector<size_t>>(strides_->begin(), strides_->end());
    }

    /**
     * @brief Fills the array with the given value.
     * Uses parallel execution for large arrays.
     */
    void fill(T value) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        if (data_->size() > 10000) {  // Threshold for parallel execution
            #pragma omp parallel for
            for (size_t i = 0; i < data_->size(); ++i) {
                (*data_)[i] = value;
            }
        } else {
            std::fill(data_->begin(), data_->end(), value);
        }
    }

    /**
     * @brief Returns an iterator to the beginning of the flat data.
     */
    auto begin() { return data_->begin(); }

    /**
     * @brief Returns an iterator to the end of the flat data.
     */
    auto end() { return data_->end(); }

    /**
     * @brief Returns a const iterator to the beginning of the flat data.
     */
    auto begin() const { return data_->begin(); }

    /**
     * @brief Returns a const iterator to the end of the flat data.
     */
    auto end() const { return data_->end(); }

    /**
     * @brief Returns a const iterator to the beginning of the flat data.
     */
    auto cbegin() const { return data_->cbegin(); }

    /**
     * @brief Returns a const iterator to the end of the flat data.
     */
    auto cend() const { return data_->cend(); }

    /**
     * @brief Returns a flattened 1D copy of the array.
     */
    ndarray<T> flatten() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        ndarray<T> result({size()});
        std::copy(data_->begin(), data_->end(), result.begin());
        return result;
    }

    /**
     * @brief Returns a 1D view of the array (zero-copy, ravel).
     */
    ndarray<T> ravel() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        // Create a 1D view with shared data
        small_vector<size_t> new_shape = {size()};
        small_vector<size_t> new_strides_vec = {1};
        auto new_strides = std::make_shared<small_vector<size_t>>(new_strides_vec);
        return ndarray<T>(new_shape, new_strides, data_);
    }

    // ---------------- Arithmetic Operators ----------------

    /**
     * @brief Element-wise addition of two ndarrays with broadcasting.
     */
    friend ndarray<T> operator+(const ndarray<T>& lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::plus<T>{});
    }

    /**
     * @brief Element-wise addition of ndarray and scalar.
     */
    friend ndarray<T> operator+(const ndarray<T>& lhs, T rhs) {
        return element_wise_op(lhs, rhs, std::plus<T>{});
    }

    /**
     * @brief Element-wise addition of scalar and ndarray.
     */
    friend ndarray<T> operator+(T lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::plus<T>{});
    }

    /**
     * @brief Element-wise subtraction of two ndarrays with broadcasting.
     */
    friend ndarray<T> operator-(const ndarray<T>& lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::minus<T>{});
    }

    /**
     * @brief Element-wise subtraction of ndarray and scalar.
     */
    friend ndarray<T> operator-(const ndarray<T>& lhs, T rhs) {
        return element_wise_op(lhs, rhs, std::minus<T>{});
    }

    /**
     * @brief Element-wise subtraction of scalar and ndarray.
     */
    friend ndarray<T> operator-(T lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::minus<T>{});
    }

    /**
     * @brief Element-wise multiplication of two ndarrays with broadcasting.
     */
    friend ndarray<T> operator*(const ndarray<T>& lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::multiplies<T>{});
    }

    /**
     * @brief Element-wise multiplication of ndarray and scalar.
     */
    friend ndarray<T> operator*(const ndarray<T>& lhs, T rhs) {
        return element_wise_op(lhs, rhs, std::multiplies<T>{});
    }

    /**
     * @brief Element-wise multiplication of scalar and ndarray.
     */
    friend ndarray<T> operator*(T lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::multiplies<T>{});
    }

    /**
     * @brief Element-wise division of two ndarrays with broadcasting.
     */
    friend ndarray<T> operator/(const ndarray<T>& lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::divides<T>{});
    }

    /**
     * @brief Element-wise division of ndarray and scalar.
     */
    friend ndarray<T> operator/(const ndarray<T>& lhs, T rhs) {
        return element_wise_op(lhs, rhs, std::divides<T>{});
    }

    /**
     * @brief Element-wise division of scalar and ndarray.
     */
    friend ndarray<T> operator/(T lhs, const ndarray<T>& rhs) {
        return element_wise_op(lhs, rhs, std::divides<T>{});
    }

    /**
     * @brief Element-wise equality comparison.
     */
    friend bool operator==(const ndarray<T>& lhs, const ndarray<T>& rhs) {
        if (lhs.shape() != rhs.shape()) return false;
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    /**
     * @brief Element-wise inequality comparison.
     */
    friend bool operator!=(const ndarray<T>& lhs, const ndarray<T>& rhs) {
        return !(lhs == rhs);
    }

    /**
     * @brief Prints a summary of the ndarray to an output stream.
     *
     * Shows shape and (for 1D arrays) the elements. For 0-D arrays, shows the scalar value.
     */
    friend std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        if (arr.shape_.empty()) return os << "ndarray(shape=(), value=" << arr() << ")";
        if (arr.shape_.size() == 1) {
            os << "[ ";
            for (size_t i = 0; i < arr.shape_[0]; ++i) os << arr[i] << " ";
            os << "]";
            return os;
        }
        os << "ndarray(shape=(";
        for (size_t i = 0; i < arr.shape_.size(); ++i)
            os << arr.shape_[i] << (i + 1 < arr.shape_.size() ? ", " : "");
        os << "))";
        return os;
    }

private:
    small_vector<size_t> shape_;                   ///< Dimensions of the array.
    std::shared_ptr<small_vector<size_t>> strides_; ///< Strides for linear indexing.
    std::shared_ptr<std::vector<T>> data_;        ///< Shared contiguous storage.

    /**
     * @brief Helper function for element-wise operations with broadcasting.
     * Uses parallel execution for large arrays.
     */
    template <typename BinaryOp>
    static ndarray<T> element_wise_op(const ndarray<T>& lhs, const ndarray<T>& rhs, BinaryOp op) {
        // Determine broadcast shape
        std::vector<size_t> broadcast_shape = broadcast_shapes(lhs.shape(), rhs.shape());
        // Broadcast both arrays to the common shape
        ndarray<T> lhs_broadcast = broadcast_to(lhs, broadcast_shape);
        ndarray<T> rhs_broadcast = broadcast_to(rhs, broadcast_shape);
        // Perform element-wise operation
        ndarray<T> result(broadcast_shape);
        if (result.size() > 10000) {
            #pragma omp parallel for
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = op(lhs_broadcast[i], rhs_broadcast[i]);
            }
        } else {
            std::transform(lhs_broadcast.begin(), lhs_broadcast.end(), rhs_broadcast.begin(), result.begin(), op);
        }
        return result;
    }

    /**
     * @brief Helper function for element-wise operations with scalar.
     * Uses parallel execution for large arrays.
     */
    template <typename BinaryOp>
    static ndarray<T> element_wise_op(const ndarray<T>& lhs, T rhs, BinaryOp op) {
        ndarray<T> result(lhs.shape());
        if (result.size() > 10000) {
            #pragma omp parallel for
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = op(lhs[i], rhs);
            }
        } else {
            std::transform(lhs.begin(), lhs.end(), result.begin(), [rhs, op](T val) { return op(val, rhs); });
        }
        return result;
    }

    /**
     * @brief Helper function for element-wise operations with scalar (lhs scalar).
     * Uses parallel execution for large arrays.
     */
    template <typename BinaryOp>
    static ndarray<T> element_wise_op(T lhs, const ndarray<T>& rhs, BinaryOp op) {
        ndarray<T> result(rhs.shape());
        if (result.size() > 10000) {
            #pragma omp parallel for
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = op(lhs, rhs[i]);
            }
        } else {
            std::transform(rhs.begin(), rhs.end(), result.begin(), [lhs, op](T val) { return op(lhs, val); });
        }
        return result;
    }

    /**
     * @brief Computes the broadcast shape of two shapes.
     */
    template <typename VecT1, typename VecT2>
    static small_vector<size_t> broadcast_shapes(const VecT1& shape1, const VecT2& shape2) {
        size_t len1 = shape1.size();
        size_t len2 = shape2.size();
        size_t max_len = std::max(len1, len2);
        small_vector<size_t> result(max_len);
        for (size_t i = 0; i < max_len; ++i) {
            size_t dim1 = (i >= len1) ? 1 : shape1[len1 - 1 - i];
            size_t dim2 = (i >= len2) ? 1 : shape2[len2 - 1 - i];
            if (dim1 != 1 && dim2 != 1 && dim1 != dim2) {
                throw std::invalid_argument("broadcast_shapes: shapes are not broadcastable");
            }
            result[max_len - 1 - i] = std::max(dim1, dim2);
        }
        return result;
    }

    /**
     * @brief Broadcasts an ndarray to a target shape.
     * Uses parallel execution for large arrays.
     */
    template <typename VecT>
    static ndarray<T> broadcast_to(const ndarray<T>& arr, const VecT& target_shape) {
        if (arr.shape() == target_shape) return arr;
        // Simple broadcast implementation for now
        ndarray<T> result(target_shape);
        // For simplicity, assume compatible shapes and copy data accordingly
        // This is a placeholder; full broadcasting logic would be more complex
        if (result.size() > 10000) {
            #pragma omp parallel for
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = arr[i % arr.size()];  // Simple repeating broadcast
            }
        } else {
            std::copy(arr.begin(), arr.end(), result.begin());
        }
        return result;
    }

    // ---------------- Slicing ----------------

    /**
     * @brief Creates a view slice of the ndarray using slice objects.
     *
     * @param slices Variadic slice objects for each dimension.
     * @return ndarray<T> A view with adjusted shape and strides.
     */
    template <typename... Slices>
    std::enable_if_t<(std::is_same_v<std::decay_t<Slices>, slice> && ...), ndarray<T>>
    operator()(Slices... slices) const {
        static_assert(sizeof...(Slices) == shape_.size(), "Number of slices must match number of dimensions");
        std::vector<slice> slice_vec = {slices...};
        return create_slice_view(slice_vec);
    }

private:
    /**
     * @brief Creates a slice view from a vector of slice objects.
     */
    ndarray<T> create_slice_view(const std::vector<slice>& slices) const {
        small_vector<size_t> new_shape;
        small_vector<size_t> new_strides;
        size_t offset = 0;

        for (size_t i = 0; i < shape_.size(); ++i) {
            const slice& s = slices[i];
            size_t start = (s.start < 0) ? shape_[i] + s.start : s.start;
            size_t stop = (s.stop < 0) ? shape_[i] + s.stop : s.stop;
            if (start >= shape_[i] || stop > shape_[i] || start > stop) {
                throw std::out_of_range("slice: invalid slice indices");
            }
            size_t length = (stop - start + s.step - 1) / s.step;  // Ceiling division
            if (length > 0) {
                new_shape.push_back(length);
                new_strides.push_back((*strides_)[i] * s.step);
                offset += start * (*strides_)[i];
            }
        }

        // Create new strides shared_ptr
        auto new_strides_ptr = std::make_shared<small_vector<size_t>>(new_strides);

        // Create view with offset in data
        // For simplicity, create a new data vector starting from offset
        // In a full implementation, this would be a proper view without copying
        auto new_data = std::make_shared<std::vector<T>>(data_->begin() + offset, data_->end());

        return ndarray<T>(new_shape, new_strides_ptr, new_data);
    }

    /**
     * @brief Initializes array storage from a given shape and fill value.
     *
     * @throws std::invalid_argument if shape is invalid.
     */
    void init_from_shape(const small_vector<size_t>& shape, T init) {
        if (std::any_of(shape.begin(), shape.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        shape_ = shape;
        size_t total = 1;
        for (size_t dim : shape) {
            if (dim > std::numeric_limits<size_t>::max() / total)
                throw std::overflow_error("ndarray: shape product overflow");
            total *= dim;
        }
        data_ = std::make_shared<std::vector<T>>();
        data_->reserve(total);
        data_->assign(total, init);
        compute_strides();
    }

    /**
     * @brief Computes row-major strides for the current shape.
     */
    void compute_strides() {
        strides_ = std::make_shared<small_vector<size_t>>(shape_.size());
        if (shape_.empty()) return;
        (*strides_).back() = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape_.size()) - 2; i >= 0; --i) {
            size_t next_stride = (*strides_)[i + 1];
            if (shape_[i + 1] > std::numeric_limits<size_t>::max() / next_stride)
                throw std::overflow_error("ndarray: stride computation overflow");
            (*strides_)[i] = next_stride * shape_[i + 1];
        }
    }

    /**
     * @brief Converts a multi-dimensional index to a flat offset.
     *
     * @throws std::invalid_argument if rank mismatch.
     * @throws std::out_of_range if any index exceeds bounds.
     */
    size_t flat_index(const std::vector<size_t>& idxs) const {
        if (idxs.size() != shape_.size())
            throw std::invalid_argument("ndarray: number of indices does not match rank");
        size_t offset = 0;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (idxs[i] >= shape_[i])
                throw std::out_of_range("ndarray: index out of bounds");
            offset += idxs[i] * (*strides_)[i];
        }
        return offset;
    }
};

/**
 * @brief Transpose an N-dimensional ndarray according to the given axes permutation.
 *
 * @param arr The input ndarray to transpose.
 * @param axes A vector specifying the new order of axes. Must be a permutation of 0 to ndim-1.
 * @return A new ndarray with transposed axes.
 *
 * @throws std::invalid_argument if axes is not a valid permutation or has wrong size.
 */
template <typename T>
ndarray<T> transpose(const ndarray<T>& arr, const std::vector<size_t>& axes) {
    if (axes.size() != arr.shape().size()) {
        throw std::invalid_argument("transpose: axes size must match number of dimensions");
    }

    // Check that axes is a valid permutation
    std::vector<bool> used(arr.shape().size(), false);
    for (size_t ax : axes) {
        if (ax >= arr.shape().size() || used[ax]) {
            throw std::invalid_argument("transpose: axes must be a valid permutation");
        }
        used[ax] = true;
    }

    // Compute new shape
    std::vector<size_t> new_shape(arr.shape().size());
    for (size_t i = 0; i < axes.size(); ++i) {
        new_shape[i] = arr.shape()[axes[i]];
    }

    // Compute new strides
    small_vector<size_t> new_strides(arr.shape().size());
    for (size_t i = 0; i < axes.size(); ++i) {
        new_strides[i] = (*arr.strides_)[axes[i]];
    }

    // Create transposed array with shared data
    auto new_strides_ptr = std::make_shared<small_vector<size_t>>(new_strides);
    return ndarray<T>(new_shape, new_strides_ptr, arr.data_ptr());
}

}  // namespace numbits
