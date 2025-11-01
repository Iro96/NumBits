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
#include <memory>
#include <tuple>

namespace numbits {

template <typename T>
class ndarray {
public:
    ndarray() noexcept : shape_(), data_(std::make_shared<std::vector<T>>()) {}

    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        init_from_shape(std::vector<size_t>(shape.begin(), shape.end()), init);
    }

    explicit ndarray(const std::vector<size_t>& shape, T init = T()) {
        init_from_shape(shape, init);
    }

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

    const std::vector<size_t>& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return data_ ? data_->size() : 0; }

    T& operator[](size_t i) {
        auto* d = data_.get();
        if (!d) throw std::logic_error("ndarray: data not initialized");
        if (i >= d->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*d)[i];
    }

    const T& operator[](size_t i) const {
        const auto* d = data_.get();
        if (!d) throw std::logic_error("ndarray: data not initialized");
        if (i >= d->size()) throw std::out_of_range("ndarray: index out of bounds");
        return (*d)[i];
    }

    template <typename... Idxs>
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

    std::vector<T>& data() {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }
    const std::vector<T>& data() const {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        return *data_;
    }

    std::shared_ptr<std::vector<T>> data_ptr() const noexcept { return data_; }

    void fill(T value) {
        if (!data_) throw std::logic_error("ndarray: data not initialized");
        std::fill(data_->begin(), data_->end(), value);
    }

    // ----------------------------
    // Fully recursive operator<<
    // ----------------------------
    friend std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        if (arr.shape_.empty()) return os << "ndarray(shape=())";

        const auto& s = arr.shape_;
        const auto* data_ptr = arr.data_->data();

        std::function<void(size_t, size_t, size_t)> print_rec;
        print_rec = [&](size_t offset, size_t dim, size_t stride) {
            if (dim == s.size() - 1) {
                os << "[ ";
                for (size_t i = 0; i < s[dim]; ++i)
                    os << data_ptr[offset + i * stride] << " ";
                os << "]";
            } else {
                os << "[";
                for (size_t i = 0; i < s[dim]; ++i) {
                    print_rec(offset + i * stride, dim + 1, stride / s[dim + 1]);
                    if (i + 1 < s[dim]) os << ", ";
                }
                os << "]";
            }
        };

        size_t total_stride = 1;
        for (size_t i = s.size() - 1; i-- > 0;)
            total_stride *= s[i + 1];

        print_rec(0, 0, total_stride);
        return os;
    }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::shared_ptr<std::vector<T>> data_;

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
