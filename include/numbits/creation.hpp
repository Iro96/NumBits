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

// Diagonal matrix creation
template<typename T>
ndarray<T> diag(const ndarray<T>& arr, int k = 0) {
    const auto& shape = arr.shape();

    // 1D array -> create diagonal matrix
    if (shape.size() == 1) {
        size_t n = shape[0];
        size_t rows = n + (k > 0 ? k : 0);
        size_t cols = n + (k < 0 ? -k : 0);

        ndarray<T> result({rows, cols});
        result.fill(T{0});

        for (size_t i = 0; i < n; ++i) {
            int r = static_cast<int>(i) + (k > 0 ? k : 0);
            int c = static_cast<int>(i) + (k < 0 ? -k : 0);
            if (r >= 0 && c >= 0 && static_cast<size_t>(r) < rows && static_cast<size_t>(c) < cols) {
                result[r * cols + c] = arr[i];
            }
        }
        return result;
    }

    else if (shape.size() == 2) {
        // 2D array -> extract diagonal
        size_t rows = shape[0];
        size_t cols = shape[1];
        
        // Determine diagonal start
        int start_row = k < 0 ? -k : 0;
        int start_col = k > 0 ? k : 0;

        // Determine diagonal length
        size_t len = 0;
        if (start_row < rows && start_col < cols) {
            len = std::min(rows - start_row, cols - start_col);
        }

        ndarray<T> result({len});
        for (size_t i = 0; i < len; ++i) {
            result[i] = arr[(start_row + i) * cols + (start_col + i)];
        }
        return result;
    }

    throw std::runtime_error("diag: input must be 1D or 2D ndarray");
}

// Vander
template<typename T>
ndarray<T> vander(const ndarray<T>& x, size_t N, bool increasing = false) {
    const auto& shape = x.shape();

    if (shape.size() != 1) {
        throw std::runtime_error("vander: input must be a 1D ndarray");
    }

    size_t M = shape[0];               // number of samples
    ndarray<T> result({M, N});
    result.fill(T{0});

    for (size_t i = 0; i < M; ++i) {
        T base = x[i];

        for (size_t j = 0; j < N; ++j) {
            size_t power;

            if (!increasing) {
                // default behavior: highest power first
                power = N - 1 - j;
            } else {
                // increasing=True: lowest power first
                power = j;
            }

            // compute base^power
            T val = T{1};
            for (size_t p = 0; p < power; ++p) {
                val *= base;
            }

            result[i * N + j] = val;
        }
    }

    return result;
}

} // namespace numbits
