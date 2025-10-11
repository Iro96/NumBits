#pragma once
#include <vector>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <sstream>

namespace nb {

template<typename T>
class ndarray {
private:
    std::vector<size_t> shape_;
    std::vector<T> data_;

public:
    ndarray() = default;

    explicit ndarray(std::vector<size_t> shape, T fill_value = T()) 
        : shape_(std::move(shape)) {
        size_t total = std::accumulate(shape_.begin(), shape_.end(), 1ul, std::multiplies<>());
        data_.assign(total, fill_value);
    }

    ndarray(std::initializer_list<T> list)
        : shape_{list.size()}, data_(list) {}

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    T& operator()(size_t i) { return data_[i]; }
    const T& operator()(size_t i) const { return data_[i]; }

    std::vector<T>& data() { return data_; }
    const std::vector<T>& data() const { return data_; }

    ndarray<T> operator+(const ndarray<T>& other) const {
        if (data_.size() != other.data_.size()) throw std::runtime_error("Shape mismatch");
        ndarray<T> out(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            out.data_[i] = data_[i] + other.data_[i];
        return out;
    }

    ndarray<T> operator*(const ndarray<T>& other) const {
        if (data_.size() != other.data_.size()) throw std::runtime_error("Shape mismatch");
        ndarray<T> out(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            out.data_[i] = data_[i] * other.data_[i];
        return out;
    }

    std::string str() const {
        std::ostringstream os;
        os << "ndarray([";
        for (size_t i = 0; i < data_.size(); ++i) {
            os << data_[i];
            if (i + 1 < data_.size()) os << ", ";
        }
        os << "])";
        return os.str();
    }

    void print() const { std::cout << str() << std::endl; }
};

template<typename T>
ndarray<T> arange(T start, T stop, T step = 1) {
    std::vector<T> data;
    for (T i = start; i < stop; i += step) data.push_back(i);
    return ndarray<T>({data.size()}, data);
}

} // namespace nb
