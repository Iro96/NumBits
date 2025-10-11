#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <algorithm>

namespace numbits {

template <typename T>
class ndarray {
public:
    ndarray() = default;

    // Enforce non-empty shape
    explicit ndarray(std::initializer_list<size_t> shape, T init = T()) {
        if (shape.size() == 0)
            throw std::invalid_argument("ndarray: shape cannot be empty");

        shape_.assign(shape.begin(), shape.end());
        const size_t total = std::accumulate(
            shape_.begin(), shape_.end(), static_cast<size_t>(1), std::multiplies<size_t>());

        data_.assign(total, init);
    }

    const std::vector<size_t>& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return data_.size(); }

    // --- Safe 2D Access ---
    T& operator()(size_t i, size_t j) {
        validate_2d_access(i, j);
        return data_[i * shape_[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        validate_2d_access(i, j);
        return data_[i * shape_[1] + j];
    }

    std::vector<T>& data() noexcept { return data_; }
    const std::vector<T>& data() const noexcept { return data_; }

    void fill(T value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        std::fill(data_.begin(), data_.end(), value);
    }

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
    std::vector<T> data_;

    void validate_2d_access(size_t i, size_t j) const {
        if (shape_.size() < 2)
            throw std::logic_error("ndarray: invalid 2D access on non-2D array");
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray: index out of bounds");
    }
};

}  // namespace nb
