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
     * The provided shape defines the size of each dimension and is used to allocate contiguous storage.
     *
     * @param shape List of dimensions; must contain at least one element and each dimension must be greater than 0.
     * @param init Value used to initialize every element (defaults to `T()`).
     *
     * @throws std::invalid_argument if `shape` is empty.
     * @throws std::invalid_argument if any dimension in `shape` is zero.
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
     * @brief Constructs an ndarray with the given shape and initializes all elements to `init`.
     *
     * The constructor validates that `shape` is not empty and that every dimension is greater than zero,
     * then allocates contiguous storage sized to the product of the shape dimensions and fills it with `init`.
     *
     * @param shape Vector of dimensions; must contain at least one element and each element must be > 0.
     * @param init Value used to initialize every element in the array (default-constructed `T()` if omitted).
     * @throws std::invalid_argument if `shape` is empty, contains a zero dimension, or other validation fails.
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
     * @brief Constructs an ndarray that wraps the provided shared data with the specified shape.
     *
     * @param shape Vector of dimension sizes; must not be empty and each dimension must be greater than 0.
     * @param data_ptr Shared pointer to a contiguous data vector whose size must equal the product of the shape dimensions; must not be null.
     *
     * @throws std::invalid_argument if `shape` is empty.
     * @throws std::invalid_argument if any dimension in `shape` is 0.
     * @throws std::invalid_argument if `data_ptr` is null.
     * @throws std::invalid_argument if `data_ptr->size()` does not equal the product of the shape dimensions.
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

    /**
 * @brief Accesses the array's shape.
 *
 * @return const std::vector<size_t>& Reference to a vector containing the size of each dimension.
 */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /**
 * @brief Get the total number of elements in the array.
 *
 * @return Total number of elements; zero if the underlying data is not initialized.
 */
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /**
     * @brief Accesses the element at the given row and column in a 2D array.
     *
     * @param i Row index (0-based).
     * @param j Column index (0-based).
     * @return Reference to the element at row `i` and column `j`.
     *
     * @throws std::logic_error if the array is not 2-dimensional.
     * @throws std::logic_error if the array is not initialized.
     * @throws std::out_of_range if `i` or `j` is outside the valid range for their dimension.
     */
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    /**
     * Accesses the element at the specified row and column in a 2D array.
     *
     * @param i Row index (zero-based).
     * @param j Column index (zero-based).
     * @return const T& Reference to the element located at (i, j).
     * @throws std::logic_error if the array is not initialized.
     * @throws std::logic_error If the array is not 2D or the underlying data is not initialized.
     * @throws std::out_of_range If `i` or `j` is outside the valid range for the corresponding dimension.
     */
    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    /**
     * @brief Accesses the underlying mutable data vector.
     *
     * @return std::vector<T>& Reference to the underlying data storage.
     *
     * @throws std::logic_error If the underlying shared data pointer is not initialized.
     */
    std::vector<T>& data() {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }
    /**
     * @brief Accesses the underlying contiguous storage as a const reference.
     *
     * Provides read-only access to the internal std::vector<T> that holds the array
     * elements.
     *
     * @returns const std::vector<T>& Const reference to the underlying data vector.
     *
     * @throws std::logic_error if the underlying data pointer is not initialized.
     */
    const std::vector<T>& data() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /**
 * Access the shared pointer to the underlying data vector.
 *
 * @return Shared pointer to the underlying std::vector<T> that stores the array elements.
 */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /**
     * @brief Fills every element of the array with the specified value.
     *
     * @throws std::logic_error if the array is not initialized.
     *
     * @param value Value to assign to each element.
     */
    void fill(T value) {
        if (!data_)
            throw std::logic_error("ndarray: data not initialized");
        std::fill(data_->begin(), data_->end(), value);
     }

    /** Pretty-print 2D arrays; for others, show shape */
    /**
     * @brief Formats an ndarray for stream output.
     *
     * Prints 2D arrays as rows enclosed in brackets, each row on its own line.
     * For arrays that are not 2D, prints a compact shape representation like `ndarray(shape=(d1, d2, ...))`.
     *
     * @param os Output stream to write to.
     * @param arr Array to format.
     * @return std::ostream& Reference to the output stream.
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
    std::vector<size_t> shape_;
    std::shared_ptr<std::vector<T>> data_;

    /**
     * @brief Validates that the array is 2-dimensional and that the given row and column indices are within bounds.
     *
     * @param i Row index.
     * @param j Column index.
     *
     * @throws std::logic_error if the array shape does not represent a 2D tensor.
     * @throws std::out_of_range if `i` or `j` is outside the valid range for their respective dimension.
     */

    void validate_2d_access(size_t i, size_t j) const {
        if (!data_)
            throw std::logic_error("ndarray: data not initialized");
        if (shape_.size() != 2)
            throw std::logic_error("ndarray: invalid 2D access on non-2D array");
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray: index out of bounds");
    }
};

}  // namespace numbits
