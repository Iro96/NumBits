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
 * Supports n-D indexing via `operator()` with variadic arguments, 1D access
 * via `operator[]`, element access, fill, and pretty printing.
 *
 * Internally stores data in a contiguous `std::vector<T>` with automatic strides.
 *
 * @tparam T Type of elements.
 */
template <typename T>
class ndarray {
public:
    // ---------------- Constructors ----------------

    /**
     * @brief Constructs an empty ndarray with shared empty storage.
     *
     * The resulting ndarray has no shape and its underlying data is an empty
     * std::vector wrapped in a shared_ptr.
     */
    ndarray()
        : data_(std::make_shared<std::vector<T>>()) {}

    /**
     * @brief Constructs an n-dimensional array with the specified shape and fills it with a value.
     *
     * @param shape List of dimensions; must contain at least one element and each dimension must be greater than 0.
     * @param init Value used to initialize every element (defaults to `T()`).
     *
     * @throws std::invalid_argument if `shape` is empty or any dimension is 0.
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(std::vector<size_t>(shape.begin(), shape.end()), init);
    }

    /**
     * @brief Constructs an ndarray with the given shape and initializes all elements to `init`.
     *
     * @param shape Vector of dimensions; must contain at least one element and each element must be > 0.
     * @param init Value used to initialize every element in the array (default-constructed `T()` if omitted).
     *
     * @throws std::invalid_argument if `shape` is empty or any dimension is 0.
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(shape, init);
    }

    /**
     * @brief Constructs an ndarray that wraps the provided shared data with the specified shape.
     *
     * @param shape Vector of dimension sizes; must not be empty and each dimension must be greater than 0.
     * @param data_ptr Shared pointer to a contiguous data vector whose size must equal the product of the shape dimensions; must not be null.
     *
     * @throws std::invalid_argument if `shape` is empty, any dimension is 0, `data_ptr` is null, or sizes mismatch.
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
     * @brief Constructs an ndarray from a shape and an initializer list of values.
     *
     * The number of elements in `values` must match the product of the shape dimensions.
     *
     * @param shape List of dimensions; must not be empty.
     * @param values Initializer list of element values.
     *
     * @throws std::invalid_argument if shape is empty, contains 0, or values size mismatch.
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

    const std::vector<size_t>& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    T& operator[](size_t i) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        if (i >= data_->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*data_)[i];
    }

    const T& operator[](size_t i) const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        if (i >= data_->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*data_)[i];
    }

    /**
     * @brief N-dimensional indexing.
     *
     * Usage: arr(i,j,k,...) where number of indices = rank (shape.size())
     *
     * @tparam Idxs Variadic size_t indices
     * @param indices List of indices
     * @return Reference to element
     * @throws std::invalid_argument if number of indices mismatch
     * @throws std::out_of_range if any index is out of bounds
     */
    template <typename... Idxs>
    T& operator()(Idxs... indices) {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...), "All indices must be size_t");
        std::vector<size_t> idxs{static_cast<size_t>(indices)...};
        return (*data_)[flat_index(idxs)];
    }

    template <typename... Idxs>
    const T& operator()(Idxs... indices) const {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...), "All indices must be size_t");
        std::vector<size_t> idxs{static_cast<size_t>(indices)...};
        return (*data_)[flat_index(idxs)];
    }

    std::vector<T>& data() {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    const std::vector<T>& data() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    void fill(T value) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        std::fill(data_->begin(), data_->end(), value);
    }

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
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::shared_ptr<std::vector<T>> data_;

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

    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        strides_.back() = 1;
        for (int i = int(shape_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }

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
