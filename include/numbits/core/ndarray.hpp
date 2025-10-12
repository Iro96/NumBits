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

namespace numbits {

/**
 * @brief Simple n-dimensional array container with shared underlying data.
 *        Supports 2D indexing, element access, fill, and pretty printing.
 *
 * @tparam T Type of elements.
 */
template <typename T>
class ndarray {
public:
    // ---------------- Constructors ----------------

    /** Default constructor */
    ndarray()
        : data_(std::make_shared<std::vector<T>>()) {}

    /**
     * @brief Construct ndarray from initializer_list (e.g., {2,3})
     * @param shape List of dimensions (cannot be empty)
     * @param init Initial value for all elements (default T())
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        if (shape.size() == 0)
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if (std::any_of(shape.begin(), shape.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");

        shape_.assign(shape.begin(), shape.end());
        const size_t total = std::accumulate(
            shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
        data_ = std::make_shared<std::vector<T>>(total, init);
    }

    /**
     * @brief Construct ndarray from std::vector<size_t> shape
     * @param shape Vector of dimensions (cannot be empty)
     * @param init Initial value for all elements (default T())
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        if (shape.empty())
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if (std::any_of(shape.begin(), shape.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");

        shape_ = shape;
        size_t total = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        data_ = std::make_shared<std::vector<T>>(total, init);
    }

    /**
     * @brief Construct ndarray from shape and shared data pointer.
     * @param shape Shape vector (must be non-empty and dimensions > 0)
     * @param data_ptr Shared pointer to existing data (must be non-null and size must match shape)
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
    }

    // ---------------- Accessors ----------------

    /** @return Shape of the array */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /** @return Total number of elements */
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /** 2D element access (safe) */
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    std::vector<T>& data() {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }
    const std::vector<T>& data() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /** @return Shared pointer to underlying data */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /** Fill all elements with a value */
    void fill(T value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        std::fill(data_->begin(), data_->end(), value);
    }

    /** Pretty-print 2D arrays; for others, show shape */
    friend std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        if (arr.shape_.size() != 2) {
            os << "ndarray(shape=(";
            for (size_t i = 0; i < arr.shape_.size(); ++i)
                os << arr.shape_[i] << (i + 1 < arr.shape_.size() ? ", " : "");
            os << "))";
            return os;
        }

        const size_t rows = arr.shape_[0];
        const size_t cols = arr.shape_[1];
        for (size_t i = 0; i < rows; ++i) {
            os << "[ ";
            for (size_t j = 0; j < cols; ++j)
                os << arr(i, j) << " ";
            os << "]\n";
        }
        return os;
    }

private:
    std::vector<size_t> shape_;
    std::shared_ptr<std::vector<T>> data_;  ///< Shared data storage

    void validate_2d_access(size_t i, size_t j) const {
        if (shape_.size() != 2)
            throw std::logic_error("ndarray: invalid 2D access on non-2D array");
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray: index out of bounds");
    }
};

}  // namespace numbits
