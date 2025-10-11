#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <numeric>

namespace numbits {

template <typename T>
class ndarray {
public:
    ndarray() = default;

    ndarray(std::initializer_list<size_t> shape, T init = T())
        : shape_(shape),
          data_(std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>()), init) {}

    const std::vector<size_t>& shape() const { return shape_; }

    size_t size() const { return data_.size(); }

    T& operator()(size_t i, size_t j) {
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray index out of bounds");
        return data_[i * shape_[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        if (i >= shape_[0] || j >= shape_[1])
            throw std::out_of_range("ndarray index out of bounds");
        return data_[i * shape_[1] + j];
    }

    std::vector<T>& data() { return data_; }
    const std::vector<T>& data() const { return data_; }

    friend std::ostream& operator<<(std::ostream& os, const ndarray& arr) {
        size_t rows = arr.shape_[0], cols = arr.shape_[1];
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
};

} // namespace nb
