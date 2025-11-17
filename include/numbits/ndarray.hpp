#pragma once

#include "types.hpp"
#include "utils.hpp"
#include <memory>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <string>
#include <type_traits>

namespace numbits {

template<typename T>
class ndarray {
public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    // 1D constructor
    ndarray(std::initializer_list<T> data) : ndarray(Shape{data.size()}, data) {}

    // Constructors
    ndarray() : shape_(), strides_(), data_(nullptr), size_(0), owns_data_(false) {}

    explicit ndarray(const Shape& shape) 
        : shape_(shape), strides_(compute_strides(shape)), 
          size_(compute_size(shape)), owns_data_(true) {
        if (size_ > 0) {
            data_ = new T[size_];
            std::fill(data_, data_ + size_, T{0});
        } else {
            data_ = nullptr;
        }
    }

    ndarray(const Shape& shape, const std::vector<T>& data) 
        : shape_(shape), strides_(compute_strides(shape)), 
          size_(compute_size(shape)), owns_data_(true) {
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

    ndarray(const Shape& shape, std::initializer_list<T> data)
        : ndarray(shape, std::vector<T>(data)) {}

    // Copy constructor
    ndarray(const ndarray& other)
        : shape_(other.shape_), strides_(other.strides_), 
          size_(other.size_), owns_data_(true) {
        if (size_ > 0) {
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        } else {
            data_ = nullptr;
        }
    }

    // Move constructor
    ndarray(ndarray&& other) noexcept
        : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
          size_(other.size_), data_(other.data_), owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
    }

    // Destructor
    ~ndarray() {
        if (owns_data_ && data_) {
            delete[] data_;
        }
    }

    // Assignment operators
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
    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    T* data() { return data_; }
    const T* data() const { return data_; }

    // Element access
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    T& at(const std::vector<size_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Number of indices does not match dimensions");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
        }
        return data_[flatten_index(indices, strides_)];
    }

    const T& at(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Number of indices does not match dimensions");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
        }
        return data_[flatten_index(indices, strides_)];
    }

    // Iterators
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }

    // Fill operations
    void fill(const T& value) {
        std::fill(data_, data_ + size_, value);
    }

    // Zero/ones creation helpers
    static ndarray zeros(const Shape& shape) {
        ndarray arr(shape);
        arr.fill(T{0});
        return arr;
    }

    static ndarray ones(const Shape& shape) {
        ndarray arr(shape);
        arr.fill(T{1});
        return arr;
    }

    static ndarray full(const Shape& shape, const T& value) {
        ndarray arr(shape);
        arr.fill(value);
        return arr;
    }

    // Create view (doesn't own data)
    ndarray create_view(const Shape& new_shape, const Strides& new_strides, T* new_data) {
        ndarray view;
        view.shape_ = new_shape;
        view.strides_ = new_strides;
        view.data_ = new_data;
        view.size_ = compute_size(new_shape);
        view.owns_data_ = false;
        return view;
    }

    // Reshape (creates a view if possible, otherwise copies)
    ndarray reshape(const Shape& new_shape) const {
        size_t new_size = compute_size(new_shape);
        if (new_size != size_) {
            throw std::runtime_error("Cannot reshape: total size mismatch");
        }
        
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

    // Flatten
    ndarray flatten() const {
        return reshape({size_});
    }

    // Convert output to binary (0 or 1)
    void convert_to_binary() {
        // Iterate through the array using the iterator
        for (auto& value : *this) {
            value = (value > 0.5f) ? 1.0f : 0.0f;
        }
    }

    // Print
    void print(std::ostream& os = std::cout) const {
        if (ndim() == 0) {
            os << data_[0];
            return;
        }
        
        print_recursive(os, 0, 0);
        os << "\nshape: " << shape_to_string(shape_);
    }

private:
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

// Type aliases
using ndarrayf = ndarray<float>;
using ndarrayd = ndarray<double>;
using ndarrayi32 = ndarray<int32_t>;
using ndarrayi64 = ndarray<int64_t>;
using ndarrayu8 = ndarray<uint8_t>;
using ndarrayu16 = ndarray<uint16_t>;
using ndarrayu32 = ndarray<uint32_t>;
using ndarrayu64 = ndarray<uint64_t>;


} // namespace numbits
