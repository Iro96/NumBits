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

template <typename T>
class ndarray {
public:
    /**
 * @brief Constructs an empty ndarray with no shape and an empty data buffer.
 *
 * Initializes the array to have an empty shape and a shared pointer to an empty
 * contiguous data vector.
 */
ndarray() noexcept : shape_(), data_(std::make_shared<std::vector<T>>()) {}

    /**
     * @brief Constructs an ndarray with the given shape and fills all elements with `init`.
     *
     * The `shape` describes the size of each dimension (in row-major order). The array is
     * allocated with the product of the shape dimensions and every element is initialized
     * to `init`.
     *
     * @param shape Dimension sizes for the array (each element > 0).
     * @param init Value used to initialize every element (defaults to value-initialized `T`).
     *
     * @throws std::invalid_argument If `shape` is empty or any dimension is zero or if allocation fails for the computed total size.
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(std::vector<size_t>(shape.begin(), shape.end()), init);
    }

    /**
     * @brief Construct an n-dimensional array with the given shape and fill value.
     *
     * @param shape Vector of dimension sizes in row-major order; must be non-empty and each size must be > 0.
     * @param init Value used to initialize all elements (defaults to `T()`).
     *
     * @throws std::invalid_argument If `shape` is empty or any dimension is zero.
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(shape, init);
    }

    /**
     * @brief Constructs an n-dimensional array that adopts the provided shape and shared data buffer.
     *
     * The constructor stores the given shape and takes ownership of the provided shared data pointer;
     * the buffer must contain exactly the product of the shape's dimensions.
     *
     * @param shape Vector of dimension sizes (one entry per axis); must be non-empty and each dimension must be greater than zero.
     * @param data_ptr Shared pointer to a contiguous data buffer whose length must equal the product of the entries in `shape`.
     *
     * @throws std::invalid_argument if `shape` is empty, any dimension is zero, `data_ptr` is null, or `data_ptr->size()` does not match the product of `shape`.
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
     * @brief Constructs an n-dimensional array from an explicit shape and a flat list of values.
     *
     * Initializes the array shape, allocates and copies the provided values into contiguous storage,
     * and computes row-major strides for multi-dimensional indexing.
     *
     * @param shape List of dimension sizes; each element is the size of the corresponding axis.
     * @param values Flat sequence of values laid out in row-major order whose length must equal the product of the dimensions in `shape`.
     *
     * @throws std::invalid_argument if `shape` is empty.
     * @throws std::invalid_argument if any dimension in `shape` is zero.
     * @throws std::invalid_argument if the number of elements in `values` does not equal the product of the dimensions in `shape`.
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

    /**
 * @brief Get the array's shape as a sequence of dimension sizes.
 *
 * @return const std::vector<size_t>& Vector containing the size of each dimension; empty if the array has no shape (default/uninitialized or scalar).
 */
const std::vector<size_t>& shape() const noexcept { return shape_; }
    /**
 * @brief Get the number of elements stored in the array.
 *
 * @return size_t Number of elements in the underlying storage, or 0 if the array is uninitialized.
 */
size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /**
     * @brief Accesses the element at the given linear index.
     *
     * Provides direct mutable access to the underlying contiguous storage at index `i`.
     *
     * @param i Linear index into the flattened array.
     * @return T& Reference to the element at index `i`.
     * @throws std::logic_error if the array storage has not been initialized.
     * @throws std::out_of_range if `i` is outside the valid range [0, size()).
     */
    T& operator[](size_t i) {
        auto* d = data_.get();
        if (!d) throw std::logic_error("ndarray: data not initialized");
        if (i >= d->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*d)[i];
    }

    /**
     * @brief Accesses the element at flat index `i` for read-only access.
     *
     * @param i Zero-based index into the underlying contiguous storage.
     * @return const T& Reference to the element at index `i`.
     * @throws std::logic_error if the array has no allocated data.
     * @throws std::out_of_range if `i` is greater than or equal to the number of elements.
     */
    const T& operator[](size_t i) const {
        const auto* d = data_.get();
        if (!d) throw std::logic_error("ndarray: data not initialized");
        if (i >= d->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*d)[i];
    }

    template <typename... Idxs>
    /**
     * @brief Accesses the element at the specified multi-dimensional indices.
     *
     * Each provided index selects the position along the corresponding dimension; the number of indices must equal the array's rank.
     *
     * @tparam Idxs... Types of the indices (must be convertible to `size_t`).
     * @param indices One index per dimension, in row-major order.
     * @return T& Reference to the element at the specified indices.
     *
     * @throws std::invalid_argument if the number of indices does not match the array rank.
     * @throws std::out_of_range if any index is outside its dimension bounds.
     */
    inline T& operator()(Idxs... indices) {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...), "All indices must be size_t");
        if (sizeof...(indices) != shape_.size())
            throw std::invalid_argument("ndarray: number of indices does not match rank");

        size_t flat_idx = 0;
        size_t i = 0;
        auto calc = [&](size_t idx) {
            if (idx >= shape_[i]) throw std::out_of_range("ndarray: index out of bounds");
            flat_idx += idx * strides_[i++];
        };
        (calc(static_cast<size_t>(indices)), ...);
        return (*data_)[flat_idx];
    }

    template <typename... Idxs>
    /**
     * Access an element by providing one index per dimension and return it by const reference.
     *
     * Each value in `indices` corresponds to the index along that dimension in order; the
     * number of provided indices must equal the array rank.
     *
     * @param indices... Indices for each axis, in dimension order.
     * @return const T& Reference to the element at the specified multi-dimensional indices.
     * @throws std::invalid_argument if the number of indices does not match the rank.
     * @throws std::out_of_range if any index is outside the bounds of its corresponding dimension.
     */
    inline const T& operator()(Idxs... indices) const {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...), "All indices must be size_t");
        if (sizeof...(indices) != shape_.size())
            throw std::invalid_argument("ndarray: number of indices does not match rank");

        size_t flat_idx = 0;
        size_t i = 0;
        auto calc = [&](size_t idx) {
            if (idx >= shape_[i]) throw std::out_of_range("ndarray: index out of bounds");
            flat_idx += idx * strides_[i++];
        };
        (calc(static_cast<size_t>(indices)), ...);
        return (*data_)[flat_idx];
    }

    /**
     * @brief Accesses the underlying contiguous storage for mutable access.
     *
     * Provides a reference to the internal std::vector holding array elements.
     *
     * @return std::vector<T>& Reference to the internal data vector.
     * @throws std::logic_error if the internal data pointer is not initialized.
     */
    std::vector<T>& data() {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }
    /**
     * @brief Access the underlying contiguous data buffer.
     *
     * @return const std::vector<T>& Reference to the internal storage vector.
     * @throws std::logic_error if the array's data has not been initialized.
     */
    const std::vector<T>& data() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /**
 * @brief Returns the shared ownership pointer to the underlying contiguous storage.
 *
 * @return std::shared_ptr<std::vector<T>> Shared pointer to the internal data vector; may be null if the array is uninitialized.
 */
std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    /**
     * @brief Overwrites every element of the array with the given value.
     *
     * Replaces all elements in the underlying storage with |value|.
     *
     * @param value Value to assign to each element.
     * @throws std::logic_error If the underlying data is not initialized.
     */
    void fill(T value) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        std::fill(data_->begin(), data_->end(), value);
    }

    // ----------------------------
    // Fully recursive operator<<
    // ----------------------------
    friend /**
     * @brief Format and write the ndarray to an output stream as nested bracketed arrays.
     *
     * The array is printed using nested square-bracket notation matching its dimensionality
     * (e.g., a 2D array prints as `[ [a b] , [c d] ]`). If the array has an empty shape,
     * the literal string `ndarray(shape=())` is written.
     *
     * @param os Output stream to write the formatted array to.
     * @param arr The ndarray to format and write.
     * @return std::ostream& Reference to the same output stream after writing.
     */
    std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        if (arr.shape_.empty()) return os << "ndarray(shape=())";

        const auto& s = arr.shape_;
        const auto& strides = arr.strides_;
        const auto* data_ptr = arr.data_->data();

        auto print_rec = [&](auto&& self, size_t offset, size_t dim) -> void {
            if (dim == s.size() - 1) {
                os << "[ ";
                for (size_t i = 0; i < s[dim]; ++i)
                    os << data_ptr[offset + i * strides[dim]] << " ";
                os << "]";
            } else {
                os << "[";
                for (size_t i = 0; i < s[dim]; ++i) {
                    self(self, offset + i * strides[dim], dim + 1);
                    if (i + 1 < s[dim]) os << ", ";
                }
                os << "]";
            }
        };

        print_rec(print_rec, 0, 0);
        return os;
    }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::shared_ptr<std::vector<T>> data_;

    /**
     * @brief Initialize the array from a given shape and fill it with a value.
     *
     * Validates the provided shape, stores it, allocates contiguous storage sized to
     * the product of the shape dimensions, fills every element with `init`, and
     * computes row-major strides.
     *
     * @param shape Vector of dimension sizes; must be non-empty and each element must be greater than 0.
     * @param init  Value used to initialize every element of the allocated storage.
     *
     * @throws std::invalid_argument if `shape` is empty or any dimension equals 0.
     */
    inline void init_from_shape(const std::vector<size_t>& shape, const T& init) {
        if (shape.empty())
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if (std::any_of(shape.begin(), shape.end(), [](size_t d){ return d == 0; }))
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        shape_ = shape;
        const size_t total = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        data_ = std::make_shared<std::vector<T>>(total, init);
        compute_strides();
    }

    /**
     * @brief Recomputes the internal strides vector to match the current shape in row-major order.
     *
     * Updates `strides_` so that each entry gives the multiplier for that dimension when
     * mapping multi-dimensional indices to a flat offset (last dimension has stride 1).
     */
    inline void compute_strides() noexcept {
        const size_t n = shape_.size();
        strides_.resize(n);
        if (n == 0) return;
        strides_[n - 1] = 1;
        for (size_t i = n - 1; i-- > 0;)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
};

}  // namespace numbits