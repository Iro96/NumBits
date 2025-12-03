/**
 * @file ndarray.hpp
 * @brief Core n-dimensional array class for NumBits.
 *
 * Defines the ndarray class template — the core multi-dimensional array container
 * for numerical computing in NumBits.
 *
 * Features:
 *   - Arbitrary N-dimensional array support
 *   - Configurable data types (float, double, int32, int64, bool, etc.)
 *   - Efficient memory management (copy, move, ownership control)
 *   - Row-major (C-style) memory layout with explicit strides
 *   - Element access via flat indexing or multi-index access
 *   - Array creation helpers (zeros, ones, full)
 *   - Shape manipulation (reshape, flatten)
 *   - STL-compatible iterators
 *   - Pretty printing with recursive formatting
 *
 * @example
 * @code
 *   ndarray<float> arr({2, 3}, {1, 2, 3, 4, 5, 6});
 *   auto zeros = ndarray<float>::zeros({3, 4});
 *   auto reshaped = arr.reshape({3, 2});
 *   arr.print();  // Pretty print
 * @endcode
 *
 * @namespace numbits
 */

#pragma once

#include "types.hpp"
#include "utils.hpp"
#include <memory>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <string>
#include <type_traits>

namespace numbits {

/**
 * @class ndarray
 * @brief N-dimensional array container for numerical computations.
 *
 * A lightweight NumPy-like array class providing:
 *   - dynamic shapes,
 *   - strides for efficient indexing,
 *   - flexible construction,
 *   - and element-wise access.
 *
 * Data is stored in contiguous row-major layout.
 *
 * @tparam T Element type of the array.
 */
template<typename T>
class ndarray {
public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    /**
     * @brief Construct a 1D ndarray from an initializer list.
     *
     * Equivalent to creating an array of shape `{data.size()}`.
     *
     * @param data Values used to initialize the array.
     */
    ndarray(std::initializer_list<T> data) : ndarray(Shape{data.size()}, data) {}

    /**
     * @brief Default constructor. Creates an empty array.
     */
    ndarray() : shape_(), strides_(), data_(nullptr), size_(0), owns_data_(false) {}

    /**
     * @brief Construct ndarray with given shape; values initialized to zero.
     *
     * @param shape Desired array shape.
     * @throws std::runtime_error If shape is invalid.
     */
    explicit ndarray(const Shape& shape) 
        : shape_(shape), strides_(compute_strides(shape)),
          size_(compute_size(shape)), owns_data_(true) 
    {
        if (size_ > 0) {
            data_ = new T[size_];
            std::fill(data_, data_ + size_, T{0});
        } else {
            data_ = nullptr;
        }
    }

    /**
     * @brief Construct ndarray from shape and a data vector.
     *
     * @param shape Desired shape of the array.
     * @param data Flat vector of elements.
     * @throws std::runtime_error If data size does not match shape.
     */
    ndarray(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), strides_(compute_strides(shape)),
          size_(compute_size(shape)), owns_data_(true) 
    {
        if (data.size() != size_) {
            throw std::runtime_error("Data size does not match shape");
        }
        if (size_ > 0) {
            data_ = new T[size_];
            std::copy(data.begin(), data.end(), data_);
        } else {
            data_ = nullptr;
        }
    }

    /**
     * @brief Construct ndarray using an initializer list for data.
     *
     * @param shape Array shape.
     * @param data Initial data.
     */
    ndarray(const Shape& shape, std::initializer_list<T> data)
        : ndarray(shape, std::vector<T>(data)) {}

    /**
     * @brief Copy constructor (deep copy).
     *
     * Allocates new memory and copies the entire array.
     */
    ndarray(const ndarray& other)
        : shape_(other.shape_), strides_(other.strides_),
          size_(other.size_), owns_data_(true)
    {
        if (size_ > 0) {
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        } else {
            data_ = nullptr;
        }
    }

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of data from another ndarray.
     */
    ndarray(ndarray&& other) noexcept
        : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
          size_(other.size_), data_(other.data_), owns_data_(other.owns_data_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
    }

    /**
     * @brief Destructor. Frees data if this instance owns it.
     */
    ~ndarray() {
        if (owns_data_ && data_) {
            delete[] data_;
        }
    }

    /**
     * @brief Copy assignment operator (deep copy).
     */
    ndarray& operator=(const ndarray& other) {
        if (this != &other) {
            if (owns_data_ && data_) {
                delete[] data_;
            }
            shape_ = other.shape_;
            strides_ = other.strides_;
            size_ = other.size_;
            owns_data_ = true;
            if (size_ > 0) {
                data_ = new T[size_];
                std::copy(other.data_, other.data_ + size_, data_);
            } else {
                data_ = nullptr;
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment operator.
     *
     * Transfers ownership of memory and invalidates the source array.
     */
    ndarray& operator=(ndarray&& other) noexcept {
        if (this != &other) {
            if (owns_data_ && data_) {
                delete[] data_;
            }
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_ = other.size_;
            data_ = other.data_;
            owns_data_ = other.owns_data_;

            other.data_ = nullptr;
            other.size_ = 0;
            other.owns_data_ = false;
        }
        return *this;
    }

    // Accessors

    /**
     * @return Shape of the array.
     */
    const Shape& shape() const { return shape_; }

    /**
     * @return Strides for each dimension.
     */
    const Strides& strides() const { return strides_; }

    /**
     * @return Total number of elements.
     */
    size_t size() const { return size_; }

    /**
     * @return Number of dimensions.
     */
    size_t ndim() const { return shape_.size(); }

    /**
     * @return Raw data pointer.
     */
    T* data() { return data_; }

    /**
     * @return Raw const data pointer.
     */
    const T* data() const { return data_; }

    // Element Access

    /**
     * @brief Access element using flat index.
     *
     * @param index Flat index into the array.
     * @throws std::out_of_range If index is invalid.
     */
    T& operator[](size_t index) {
        if (index >= size_) throw std::out_of_range("Index out of range");
        return data_[index];
    }

    /**
     * @brief Const flat-index access.
     */
    const T& operator[](size_t index) const {
        if (index >= size_) throw std::out_of_range("Index out of range");
        return data_[index];
    }

    /**
     * @brief Multi-dimensional indexed element access.
     *
     * @param indices Vector of indices, one per dimension.
     * @throws std::runtime_error For incorrect number of indices.
     * @throws std::out_of_range For bounds violations.
     */
    T& at(const std::vector<size_t>& indices) {
        if (indices.size() != shape_.size())
            throw std::runtime_error("Number of indices does not match dimensions");
        for (size_t i = 0; i < indices.size(); ++i)
            if (indices[i] >= shape_[i]) throw std::out_of_range("Index out of range");

        return data_[flatten_index(indices, strides_)];
    }

    /**
     * @brief Const version of at().
     */
    const T& at(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size())
            throw std::runtime_error("Number of indices does not match dimensions");
        for (size_t i = 0; i < indices.size(); ++i)
            if (indices[i] >= shape_[i]) throw std::out_of_range("Index out of range");

        return data_[flatten_index(indices, strides_)];
    }

    // Iterators

    /** @return Iterator to beginning. */
    iterator begin() { return data_; }

    /** @return Iterator past end. */
    iterator end() { return data_ + size_; }

    /** @return Const iterator to beginning. */
    const_iterator begin() const { return data_; }

    /** @return Const iterator past end. */
    const_iterator end() const { return data_ + size_; }

    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }

    // Fill Operations

    /**
     * @brief Fill the array with a constant value.
     *
     * @param value Value to assign to all elements.
     */
    void fill(const T& value) {
        std::fill(data_, data_ + size_, value);
    }

    // Factory Methods

    /**
     * @brief Create an array filled with zeros.
     */
    static ndarray zeros(const Shape& shape) {
        ndarray arr(shape);
        arr.fill(T{0});
        return arr;
    }

    /**
     * @brief Create an array filled with ones.
     */
    static ndarray ones(const Shape& shape) {
        ndarray arr(shape);
        arr.fill(T{1});
        return arr;
    }

    /**
     * @brief Create an array filled with a specific value.
     *
     * @param shape Array shape.
     * @param value Value for all elements.
     */
    static ndarray full(const Shape& shape, const T& value) {
        ndarray arr(shape);
        arr.fill(value);
        return arr;
    }

    // Views & Reshape

    /**
     * @brief Create a non-owning view of data with custom shape and strides.
     *
     * @param new_shape Shape of the view.
     * @param new_strides Strides for the view.
     * @param new_data Pointer to the underlying data.
     * @return A new ndarray sharing the same memory (does not own data).
     */
    ndarray create_view(const Shape& new_shape, const Strides& new_strides, T* new_data) {
        ndarray view;
        view.shape_ = new_shape;
        view.strides_ = new_strides;
        view.data_ = new_data;
        view.size_ = compute_size(new_shape);
        view.owns_data_ = false;
        return view;
    }

    /**
     * @brief Reshape array into new dimensions.
     *
     * Produces a deep copy; does not reuse original memory.
     *
     * @param new_shape Desired shape.
     * @throws std::runtime_error If total size differs.
     */
    ndarray reshape(const Shape& new_shape) const {
        size_t new_size = compute_size(new_shape);
        if (new_size != size_)
            throw std::runtime_error("Cannot reshape: total size mismatch");

        Strides new_strides = compute_strides(new_shape);
        ndarray result;
        result.shape_ = new_shape;
        result.strides_ = new_strides;
        result.size_ = new_size;
        result.owns_data_ = true;

        result.data_ = new T[size_];
        std::copy(data_, data_ + size_, result.data_);

        return result;
    }

    /**
     * @brief Flatten the array into 1D.
     *
     * @return A reshaped copy with shape `{size_}`.
     */
    ndarray flatten() const {
        return reshape({size_});
    }

    // Miscellaneous Ops

    /**
     * @brief Convert array to binary form (values > 0.5 become 1, else 0).
     *
     * Useful for thresholding operations.
     */
    void convert_to_binary() {
        for (auto& value : *this) {
            value = (value > 0.5f) ? 1.0f : 0.0f;
        }
    }

    /**
     * @brief Pretty print array with recursive formatting.
     *
     * @param os Output stream (default: std::cout).
     */
    void print(std::ostream& os = std::cout) const {
        if (ndim() == 0) {
            os << data_[0];
            return;
        }
        
        print_recursive(os, 0, 0);
        os << "\nshape: " << shape_to_string(shape_);
    }

private:

    /**
     * @brief Recursive pretty-printer helper.
     */
    void print_recursive(std::ostream& os, size_t dim, size_t offset) const {
        if (dim == ndim() - 1) {
            os << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0) os << ", ";
                os << data_[offset + i];
            }
            os << "]";
        } else {
            os << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0) os << ",\n" << std::string(dim + 1, ' ');
                print_recursive(os, dim + 1, offset + i * strides_[dim]);
            }
            os << "]";
        }
    }

    Shape shape_;
    Strides strides_;
    T* data_;
    size_t size_;
    bool owns_data_;
};

// Type aliases for convenience
using ndarrayf   = ndarray<float>;
using ndarrayd   = ndarray<double>;
using ndarrayi32 = ndarray<int32_t>;
using ndarrayi64 = ndarray<int64_t>;
using ndarrayu8  = ndarray<uint8_t>;
using ndarrayu16 = ndarray<uint16_t>;
using ndarrayu32 = ndarray<uint32_t>;
using ndarrayu64 = ndarray<uint64_t>;

} // namespace numbits
