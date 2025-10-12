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
 * @brief N-dimensional array class similar to NumPy ndarray.
 * 
 * Supports basic initialization, 2D access, fill, and printing.
 * Can share underlying data between reshaped or expanded arrays.
 * 
 * @tparam T Data type of the array elements.
 */
template <typename T>
class ndarray {
public:
    /**
     * @brief Default constructor creates an empty ndarray.
     */
    ndarray() = default;

    /**
     * @brief Construct an ndarray with the given shape and initial value.
     * 
     * @param shape Initial shape of the array as initializer_list.
     * @param init Value to fill the array (default-constructed if omitted).
     * @throws std::invalid_argument if shape is empty.
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(std::vector<size_t>(shape), init);
    }

    /**
     * @brief Construct an ndarray from a vector shape and initial value.
     * 
     * @param shape Vector of dimensions.
     * @param init Value to fill (default T()).
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(shape, init);
    }

    /**
     * @brief Construct an ndarray sharing underlying data.
     * 
     * Used internally for reshape, expand_dims, and squeeze operations.
     * 
     * @param shape New shape of the array.
     * @param data Shared pointer to underlying data vector.
     * @param offset Starting index in shared data.
     */
    ndarray(const std::vector<size_t>& shape, std::shared_ptr<std::vector<T>> data, size_t offset = 0)
        : shape_(shape), strides_(compute_strides(shape)), offset_(offset), data_(data) {}

    /**
     * @brief Get the shape of the array.
     * @return Reference to vector of dimensions.
     */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /**
     * @brief Get the total number of elements.
     * @return Number of elements in the array.
     */
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /**
     * @brief Access 2D element (row i, column j) for read/write.
     * @param i Row index.
     * @param j Column index.
     * @return Reference to element.
     * @throws std::logic_error if array is not 2D.
     * @throws std::out_of_range if indices are out of bounds.
     */
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return (*data_)[offset_ + i * shape_[1] + j];
    }

    /**
     * @brief Access 2D element (row i, column j) for reading.
     */
    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return (*data_)[offset_ + i * shape_[1] + j];
    }

    /**
     * @brief Get reference to underlying data vector.
     * @return Reference to std::vector<T>.
     */
    std::vector<T>& data() noexcept { return *data_; }

    /**
     * @brief Get const reference to underlying data vector.
     */
    const std::vector<T>& data() const noexcept { return *data_; }

    /**
     * @brief Fill the entire array with a given value.
     * @param value Value to fill.
     */
    void fill(T value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        std::fill(data()->begin(), data()->end(), value);
    }

    /**
     * @brief Stream output for printing 2D arrays or shapes.
     */
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
    std::vector<size_t> shape_;                   ///< Shape of the array
    std::vector<size_t> strides_;                 ///< Strides for internal calculations
    size_t offset_ = 0;                           ///< Offset for shared data
    std::shared_ptr<std::vector<T>> data_;       ///< Underlying data storage

    void init_from_shape(const std::vector<size_t>& shape, T init) {
        if (shape.empty())
            throw std::invalid_argument("ndarray: shape cannot be empty");
        shape_ = shape;
        strides_ = compute_strides(shape_);
        size_t total = std::accumulate(shape_.begin(), shape_.end(), static_cast<size_t>(1),
                                       std::multiplies<size_t>());
        data_ = std::make_shared<std::vector<T>>(total, init);
    }

    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }

    void validate_2d_access(size_t i, size_t j) const {
        if (shape_.size() != 2)
            throw std::logic_error("ndarray: invalid 2D access on non-2D array");
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray: index out of bounds");
    }
};

}  // namespace numbits
