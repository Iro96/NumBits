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
         * @brief Constructs an empty ndarray with a shared, empty data vector.
         *
         * Initializes the shape to an empty vector and allocates a new shared_ptr
         * to an empty underlying data vector.
         */
    ndarray()
        : data_(std::make_shared<std::vector<T>>()) {}

    /**
     * @brief Constructs an ndarray with the specified shape and initializes all elements to the given value.
     *
     * @param shape Sequence of dimension sizes (must not be empty; each dimension must be greater than 0).
     * @param init Value used to initialize every element in the array (default-constructed `T` if omitted).
     *
     * @throw std::invalid_argument if `shape` is empty or any dimension size is 0.
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
     * @brief Constructs an ndarray with the specified shape and fills every element with `init`.
     *
     * @param shape Vector of dimensions; must be non-empty and each dimension must be greater than 0.
     * @param init Value to initialize every element with (default-constructed if omitted).
     *
     * @throws std::invalid_argument if `shape` is empty or any dimension in `shape` is 0.
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
     * @brief Construct an ndarray that shares the provided data buffer with the given shape.
     *
     * @param shape Vector of dimension sizes; must be non-empty and each dimension must be greater than 0.
     * @param data_ptr Shared pointer to the underlying element storage; must be non-null and its size must equal the product of the dimensions in `shape`.
     *
     * @throws std::invalid_argument if `shape` is empty.
     * @throws std::invalid_argument if any dimension in `shape` is 0.
     * @throws std::invalid_argument if `data_ptr` is null.
     * @throws std::invalid_argument if `data_ptr->size()` does not equal the product of `shape` dimensions.
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
 * @brief The array's shape as a vector of dimension sizes.
 *
 * @return const std::vector<size_t>& Reference to the shape vector where each element is the size of the corresponding dimension.
 */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /**
 * Total number of elements in the array.
 *
 * @return Total number of elements; `0` if the array has no underlying data.
 */
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /**
     * @brief Accesses the element at the specified row and column and returns it by reference.
     *
     * @param i Zero-based row index.
     * @param j Zero-based column index.
     * @return T& Reference to the element at (i, j).
     *
     * @throws std::logic_error If the array is not 2-dimensional.
     * @throws std::out_of_range If `i` or `j` is outside the valid index range for their dimension.
     */
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    /**
     * @brief Accesses the element at the specified 2D index as a const reference.
     *
     * @param i Row index (0-based).
     * @param j Column index (0-based).
     * @return const T& Reference to the element at row `i` and column `j`.
     * @throws std::logic_error If the array is not 2-dimensional.
     * @throws std::out_of_range If `i` or `j` is outside the valid range for their dimension.
     */
    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return (*data_)[i * shape_[1] + j];
    }

    /**
 * Provides access to the underlying contiguous storage vector.
 *
 * @return Reference to the underlying vector that stores the array's elements.
 */
    std::vector<T>& data() noexcept { return *data_; }
    /**
 * @brief Returns a const reference to the underlying contiguous storage.
 *
 * @return const std::vector<T>& Reference to the internal vector holding the array's elements.
 */
const std::vector<T>& data() const noexcept { return *data_; }

    /**
 * Get a shared pointer to the underlying contiguous storage.
 *
 * @return std::shared_ptr<std::vector<T>> Shared pointer to the underlying data vector; may be null if the array has no allocated storage.
 */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /**
     * @brief Assigns the given value to every element in the array.
     *
     * @param value Value to write into each element.
     */
    void fill(T value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        std::fill(data_->begin(), data_->end(), value);
    }

    /** Pretty-print 2D arrays; for others, show shape */
    friend /**
     * @brief Formats an ndarray for insertion into an output stream.
     *
     * Pretty-prints 2D arrays as rows of elements enclosed in square brackets; for non-2D arrays prints the shape as `ndarray(shape=(d1, d2, ...))`.
     *
     * @param os Output stream to write to.
     * @param arr ndarray to format.
     * @return std::ostream& The same output stream passed in `os`.
     */
    std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
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
    std::shared_ptr<std::vector<T>> data_;  /**
     * @brief Validate that the array is 2D and the given row and column indices are within bounds.
     *
     * @param i Row index to validate.
     * @param j Column index to validate.
     * @throws std::logic_error If the underlying shape does not represent a 2D array.
     * @throws std::out_of_range If `i` or `j` is outside the valid range for their respective dimension.
     */

    void validate_2d_access(size_t i, size_t j) const {
        if (shape_.size() != 2)
            throw std::logic_error("ndarray: invalid 2D access on non-2D array");
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray: index out of bounds");
    }
};

}  // namespace numbits