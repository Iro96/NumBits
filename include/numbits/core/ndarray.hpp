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

namespace numbits {

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
     * @param shape List of dimensions; must not be empty and each > 0.
     * @param init  Value used to initialize all elements (defaults to `T()`).
     *
     * @throws std::invalid_argument if `shape` is empty or any dimension is 0.
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(std::vector<size_t>(shape.begin(), shape.end()), init);
    }

    /**
     * @brief Constructs an n-dimensional array with the given shape and fills it with a value.
     *
     * @param shape Vector of dimension sizes; must not be empty and each > 0.
     * @param init  Value used to initialize all elements (defaults to `T()`).
     *
     * @throws std::invalid_argument if `shape` is empty or any dimension is 0.
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(shape, init);
    }

    /**
     * @brief Constructs an ndarray that wraps an existing shared data buffer.
     *
     * @param shape     Dimensions of the array.
     * @param data_ptr  Shared pointer to contiguous vector; must match shape size.
     *
     * @throws std::invalid_argument if shape or data pointer is invalid.
     */
    ndarray(const std::vector<size_t>& shape, std::shared_ptr<std::vector<T>> data_ptr)
        : shape_(shape), data_(std::move(data_ptr)) {
        if (shape_.empty())
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if (std::any_of(shape_.begin(), shape_.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        if (!data_)
            throw std::invalid_argument("ndarray: data_ptr cannot be null");
        const size_t expected = std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
        if (data_->size() != expected)
            throw std::invalid_argument("ndarray: data size does not match shape");
        compute_strides();
    }

    /**
     * @brief Constructs an ndarray from a shape and initializer list of values.
     *
     * The number of values must exactly match the product of the shape dimensions.
     *
     * @param shape  List of dimension sizes.
     * @param values List of values to initialize the data.
     *
     * @throws std::invalid_argument if sizes do not match.
     */
    ndarray(std::initializer_list<size_t> shape, std::initializer_list<T> values) {
        shape_.assign(shape.begin(), shape.end());
        if (shape_.empty())
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if (std::any_of(shape_.begin(), shape_.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");

        const size_t total = std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
        if (values.size() != total)
            throw std::invalid_argument("ndarray: number of values does not match shape size");

        data_ = std::make_shared<std::vector<T>>(values.begin(), values.end());
        compute_strides();
    }

    // ---------------- Accessors ----------------

    /**
     * @brief Returns the array shape.
     */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

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
            flat_idx += idx * strides_[i];
            i++;
        };

        (calculate_offset(static_cast<size_t>(indices)), ...);

        return (*data_)[flat_idx];
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
            flat_idx += idx * strides_[i];
            ++i;
        };
    
        (calculate_offset(static_cast<size_t>(indices)), ...);
        return (*data_)[flat_idx];
    }

    /**
     * @brief Returns reference to internal data vector.
     * @warning External resizing invalidates shape/stride consistency.
     */
    std::vector<T>& data() {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /**
     * @brief Returns const reference to internal data vector.
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
     * @brief Fills the array with the given value.
     */
    void fill(T value) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        std::fill(data_->begin(), data_->end(), value);
    }

    /**
     * @brief Prints a summary of the ndarray to an output stream.
     *
     * Shows shape and (for 1D arrays) the elements.
     */
    friend std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        if (arr.shape_.empty()) return os << "ndarray(shape=())";
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
    std::vector<size_t> shape_;                   ///< Dimensions of the array.
    std::vector<size_t> strides_;                 ///< Strides for linear indexing.
    std::shared_ptr<std::vector<T>> data_;        ///< Shared contiguous storage.

    /**
     * @brief Initializes array storage from a given shape and fill value.
     *
     * @throws std::invalid_argument if shape is invalid.
     */
    void init_from_shape(const std::vector<size_t>& shape, T init) {
        if (shape.empty())
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if (std::any_of(shape.begin(), shape.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        shape_ = shape;
        size_t total = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        data_ = std::make_shared<std::vector<T>>(total, init);
        compute_strides();
    }

    /**
     * @brief Computes row-major strides for the current shape.
     */
    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        strides_.back() = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
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
            offset += idxs[i] * strides_[i];
        }
        return offset;
    }
};

}  // namespace numbits
