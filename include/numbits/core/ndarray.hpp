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
 * 
 * @tparam T Type of elements stored in the array.
 */
template <typename T>
class ndarray {
public:
    ndarray() = default;

    /**
     * @brief Construct an ndarray with given shape, optionally initializing values.
     * @param shape List of dimensions (cannot be empty).
     * @param init Initial value for all elements (default T()).
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        if (shape.size() == 0)
            throw std::invalid_argument("ndarray: shape cannot be empty");

        shape_.assign(shape.begin(), shape.end());
        const size_t total = std::accumulate(
            shape_.begin(), shape_.end(), static_cast<size_t>(1), std::multiplies<size_t>());

        data_ = std::make_shared<std::vector<T>>(total, init);
    }

    /**
     * @brief Construct ndarray from existing shared data and shape.
     *        Used internally for reshape / expand_dims without copying.
     */
    ndarray(const std::vector<size_t>& shape, std::shared_ptr<std::vector<T>> data_ptr)
        : shape_(shape), data_(data_ptr) {}

    /** @return Shape of the array */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /** @return Total number of elements */
    size_t size() const noexcept { return data_->size(); }

    /** 
     * @brief 2D element access (row, col) with bounds check 
     * @throws logic_error if array is not 2D
     */
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    /** @return Reference to underlying vector of data */
    std::vector<T>& data() noexcept { return *data_; }
    const std::vector<T>& data() const noexcept { return *data_; }

    /**
     * @brief Get shared pointer to underlying data (for reshaping / broadcasting).
     */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /**
     * @brief Fill the array with a value
     * @param value Value to fill
     */
    void fill(T value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        std::fill(data_->begin(), data_->end(), value);
    }

    /** @brief Stream output (only pretty prints 2D arrays) */
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
    std::shared_ptr<std::vector<T>> data_;  ///< Underlying data storage

    void validate_2d_access(size_t i, size_t j) const {
        if (shape_.size() != 2)
            throw std::logic_error("ndarray: invalid 2D access on non-2D array");
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray: index out of bounds");
    }
};

} // namespace numbits
