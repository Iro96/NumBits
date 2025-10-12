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
    ndarray() = default;

    /**
     * @brief Construct an ndarray with given shape and optional initial value.
     * @param shape List of dimension sizes (cannot be empty).
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
     * @brief Construct ndarray with existing shared data (used internally for reshape / views)
     */
    ndarray(const std::vector<size_t>& shape, std::shared_ptr<std::vector<T>> data_ptr)
        : shape_(shape), data_(data_ptr) {}

    /** @return Shape of the array */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /** @return Total number of elements */
    size_t size() const noexcept { return data_->size(); }

    // --- 2D Safe Access ---
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    /** @return Reference to underlying vector */
    std::vector<T>& data() noexcept { return *data_; }
    const std::vector<T>& data() const noexcept { return *data_; }

    /** @return Shared pointer to underlying data (for views / reshape) */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /** Fill all elements with a value */
    void fill(T value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        std::fill(data_->begin(), data_->end(), value);
    }

    /** Pretty-print 2D arrays; for others, just show shape */
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
