#pragma once

#include "ndarray.hpp"
#include <vector>
#include <stdexcept>

namespace numbits {

// NumPy-style arange
template<typename T>
ndarray<T> arange(T start, T stop, T step) {
    if (step == T{0}) {
        throw std::runtime_error("arange step cannot be zero");
    }

    std::vector<T> data;
    T value = start;
    if ((step > T{0} && start < stop) || (step < T{0} && start > stop)) {
        while ((step > T{0}) ? (value < stop) : (value > stop)) {
            data.push_back(value);
            value += step;
        }
    }

    return ndarray<T>({data.size()}, data);
}

template<typename T>
ndarray<T> arange(T stop) {
    return arange<T>(T{0}, stop, T{1});
}

template<typename T>
ndarray<T> arange(T start, T stop) {
    return arange<T>(start, stop, T{1});
}

// NumPy-style linspace
template<typename T>
ndarray<T> linspace(T start, T stop, size_t num, bool endpoint = true) {
    if (num == 0) {
        return ndarray<T>({0});
    }

    std::vector<T> data(num);
    if (num == 1) {
        data[0] = start;
        return ndarray<T>({1}, data);
    }

    T step;
    if (endpoint) {
        step = (stop - start) / static_cast<T>(num - 1);
    } else {
        step = (stop - start) / static_cast<T>(num);
    }

    for (size_t i = 0; i < num; ++i) {
        data[i] = start + static_cast<T>(i) * step;
    }

    if (endpoint) {
        data.back() = stop;
    }

    return ndarray<T>({num}, data);
}

// Identity matrix creation similar to NumPy's eye
template<typename T>
ndarray<T> eye(size_t n, size_t m = 0, int k = 0) {
    if (m == 0) {
        m = n;
    }
    ndarray<T> result({n, m});
    result.fill(T{0});

    for (size_t row = 0; row < n; ++row) {
        int col = static_cast<int>(row) + k;
        if (col >= 0 && static_cast<size_t>(col) < m) {
            result[row * m + static_cast<size_t>(col)] = T{1};
        }
    }
    return result;
}

} // namespace numbits
