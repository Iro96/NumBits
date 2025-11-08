#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace numbits {

/**
 * @brief Multi‑dimensional array class template.
 *
 * The ndarray class holds data in a contiguous 1D vector (via shared_ptr)
 * but allows access with N indices and keeps track of shape and strides.
 * @tparam T The element type.
 */
template<typename T>
class ndarray {
public:
    /**
     * @brief Default constructor. Creates an empty array (shape_ is empty, data_ initialized to empty vector).
     */
    ndarray() noexcept
        : shape_(), data_(std::make_shared<std::vector<T>>()) {}

    /**
     * @brief Constructor taking shape as initializer list and optional initial value.
     *
     * @param shape An initializer_list specifying the extents of each dimension.
     * @param init Value to initialize each element with (default‐constructed T if not specified).
     *
     * This will call init_from_shape() internally.
     */
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(std::vector<size_t>(shape.begin(), shape.end()), init);
    }

    /**
     * @brief Constructor taking shape vector and optional initial value.
     *
     * @param shape A std::vector<size_t> specifying the extents of each dimension.
     * @param init Value to initialize each element with (default‐constructed T if not specified).
     *
     * Internally calls init_from_shape().
     */
    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(shape, init);
    }

    /**
     * @brief Constructor taking shape vector and externally allocated data pointer.
     *
     * @param shape A vector of extents for each dimension. Must not be empty, and no dimension may be zero.
     * @param data_ptr A shared_ptr pointing to a vector<T> containing the data. The size of the data vector must equal the product of shape elements.
     *
     * Throws std::invalid_argument if shape is invalid or data_ptr is null or size mismatch.
     */
    ndarray(const std::vector<size_t>& shape,
            std::shared_ptr<std::vector<T>> data_ptr)
        : shape_(shape), data_(std::move(data_ptr))
    {
        if ( shape_.empty() )
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if ( std::any_of(shape_.begin(), shape_.end(),
                         [](size_t d){ return d == 0; }) )
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        if ( ! data_ )
            throw std::invalid_argument("ndarray: data_ptr cannot be null");
        const size_t expected = std::accumulate(shape_.begin(), shape_.end(),
                                                size_t{1}, std::multiplies<>());
        if ( data_->size() != expected )
            throw std::invalid_argument("ndarray: data size does not match shape");
        compute_strides();
    }

    /**
     * @brief Constructor from shape initializer list and a list of values.
     *
     * @param shape An initializer_list specifying the extents of each dimension.
     * @param values An initializer_list of values (flattened) whose size must equal the product of shape extents.
     *
     * Throws std::invalid_argument if shape is empty, dimensions zero, or values size mismatch.
     */
    ndarray(std::initializer_list<size_t> shape,
            std::initializer_list<T> values)
    {
        shape_.assign(shape.begin(), shape.end());
        if ( shape_.empty() )
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if ( std::any_of(shape_.begin(), shape_.end(),
                         [](size_t d){ return d == 0; }) )
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        const size_t total = std::accumulate(shape_.begin(), shape_.end(),
                                             size_t{1}, std::multiplies<>());
        if ( values.size() != total )
            throw std::invalid_argument("ndarray: number of values does not match shape size");
        data_ = std::make_shared<std::vector<T>>(values.begin(), values.end());
        compute_strides();
    }

    /**
     * @brief Returns the shape of the ndarray (extents of each dimension).
     * @return A const reference to the vector<size_t> shape_.
     */
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /**
     * @brief Returns the total number of elements in the array.
     * @return The size (i.e., product of dimension extents).
     */
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    /**
     * @brief 1D indexing operator (flat indexing).
     * @param i The zero‐based index into the flattened data vector.
     * @return Reference to the element at position i.
     *
     * Throws std::logic_error if data_ not initialized; std::out_of_range if i ≥ size().
     */
    T& operator[](size_t i) {
        auto* d = data_.get();
        if ( ! d ) throw std::logic_error("ndarray: data not initialized");
        if ( i >= d->size() ) throw std::out_of_range("ndarray: index out of bounds");
        return (*d)[i];
    }

    /**
     * @brief Const version of 1D indexing operator.
     * @param i The zero‐based index into the flattened data vector.
     * @return Const reference to the element at position i.
     *
     * Throws std::logic_error if data_ not initialized; std::out_of_range if i ≥ size().
     */
    const T& operator[](size_t i) const {
        const auto* d = data_.get();
        if ( ! d ) throw std::logic_error("ndarray: data not initialized");
        if ( i >= d->size() ) throw std::out_of_range("ndarray: index out of bounds");
        return (*d)[i];
    }

    /**
     * @brief Multi‐dimensional indexing operator.
     *
     * Allows indexing with N indices corresponding to each dimension.
     * Example: arr(i,j,k) for a 3D array.
     *
     * @tparam Idxs Parameter pack of index types (must convert to size_t).
     * @param indices A sequence of indices, one per dimension.
     * @return Reference to the element at the given multi‐index.
     *
     * @throws std::invalid_argument if the number of indices ≠ rank (shape_.size()).
     * @throws std::out_of_range if any index ≥ the size of its dimension.
     */
    template<typename... Idxs>
    inline T& operator()(Idxs... indices) {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...),
                      "All indices must be size_t");
        if ( sizeof...(indices) != shape_.size() )
            throw std::invalid_argument("ndarray: number of indices does not match rank");
        size_t flat_idx = 0;
        size_t i = 0;
        auto calc = [&](size_t idx) {
            if ( idx >= shape_[i] )
                throw std::out_of_range("ndarray: index out of bounds");
            flat_idx += idx * strides_[i++];
        };
        (calc(static_cast<size_t>(indices)), ...);
        return (*data_)[flat_idx];
    }

    /**
     * @brief Const version of multi‐dimensional indexing operator.
     * @tparam Idxs Parameter pack of index types.
     * @param indices A sequence of indices, one per dimension.
     * @return Const reference to the element at the given multi‐index.
     *
     * @throws std::invalid_argument if the number of indices ≠ rank.
     * @throws std::out_of_range if any index ≥ size of its dimension.
     */
    template<typename... Idxs>
    inline const T& operator()(Idxs... indices) const {
        static_assert((std::is_convertible_v<Idxs, size_t> && ...),
                      "All indices must be size_t");
        if ( sizeof...(indices) != shape_.size() )
            throw std::invalid_argument("ndarray: number of indices does not match rank");
        size_t flat_idx = 0;
        size_t i = 0;
        auto calc = [&](size_t idx) {
            if ( idx >= shape_[i] )
                throw std::out_of_range("ndarray: index out of bounds");
            flat_idx += idx * strides_[i++];
        };
        (calc(static_cast<size_t>(indices)), ...);
        return (*data_)[flat_idx];
    }

    /**
     * @brief Returns a reference to the underlying data vector.
     * @return Reference to the std::vector<T> storing the elements.
     *
     * @throws std::logic_error if data_ not initialized.
     */
    std::vector<T>& data() {
        if ( ! data_ ) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /**
     * @brief Const version of data() accessor.
     * @return Const reference to the underlying std::vector<T>.
     *
     * @throws std::logic_error if data_ not initialized.
     */
    const std::vector<T>& data() const {
        if ( ! data_ ) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    /**
     * @brief Returns the shared_ptr to the underlying data vector.
     * @return shared_ptr<std::vector<T>> pointing to the data.
     */
    std::shared_ptr<std::vector<T>> data_ptr() const noexcept {
        return data_;
    }

    /**
     * @brief Fill all elements of the array with the given value.
     * @param value The value to assign to each element.
     *
     * @throws std::logic_error if data_ not initialized.
     */
    void fill(T value) {
        if ( ! data_ ) throw std::logic_error("ndarray: data not initialized");
        std::fill(data_->begin(), data_->end(), value);
    }

    // ----------------------------
    // Fully recursive operator<<
    // ----------------------------
    /**
     * @brief Output stream insertion operator.
     * @param os Output stream.
     * @param arr The ndarray to print.
     * @return Reference to output stream for chaining.
     *
     * Prints the multi‐dimensional array in nested bracket form.
     */
    friend std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        if ( arr.shape_.empty() ) return os << "ndarray(shape=())";
        const auto& s = arr.shape_;
        const auto& strides = arr.strides_;
        const auto* data_ptr = arr.data_->data();
        auto print_rec = [&](auto&& self, size_t offset, size_t dim) -> void {
            if ( dim == s.size() - 1 ) {
                os << "[ ";
                for (size_t i = 0; i < s[dim]; ++i)
                    os << data_ptr[offset + i * strides[dim]] << " ";
                os << "]";
            }
            else {
                os << "[";
                for (size_t i = 0; i < s[dim]; ++i) {
                    self(self, offset + i * strides[dim], dim + 1);
                    if ( i + 1 < s[dim] ) os << ", ";
                }
                os << "]";
            }
        };
        print_rec(print_rec, 0, 0);
        return os;
    }

private:
    std::vector<size_t> shape_;            ///< Extents of each dimension
    std::vector<size_t> strides_;          ///< Strides for each dimension (flat index multiplier)
    std::shared_ptr<std::vector<T>> data_; ///< Shared pointer to the underlying data vector

    /**
     * @brief Internal initializer: set shape and allocate data with given init value.
     *
     * @param shape A vector of extents for each dimension. Must not be empty and no dimension may be zero.
     * @param init The initial fill value for each element.
     *
     * Throws std::invalid_argument if shape invalid.
     */
    inline void init_from_shape(const std::vector<size_t>& shape, const T& init) {
        if ( shape.empty() )
            throw std::invalid_argument("ndarray: shape cannot be empty");
        if ( std::any_of(shape.begin(), shape.end(),
                         [](size_t d){ return d == 0; }) )
            throw std::invalid_argument("ndarray: shape dimensions must be > 0");
        shape_ = shape;
        const size_t total = std::accumulate(shape.begin(), shape.end(),
                                             size_t{1}, std::multiplies<>());
        data_ = std::make_shared<std::vector<T>>(total, init);
        compute_strides();
    }

    /**
     * @brief Compute the strides_ vector from shape_.
     *
     * Strides are calculated such that for dimension i:
     *    flat_index += index[i] * strides_[i]
     * and the last dimension has stride = 1.
     */
    inline void compute_strides() noexcept {
        const size_t n = shape_.size();
        strides_.resize(n);
        if ( n == 0 ) return;
        strides_[n - 1] = 1;
        for (size_t i = n - 1; i-- > 0; ) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
};

} // namespace numbits
