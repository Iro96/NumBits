/**
 * @file creation.hpp
 * @brief Array creation functions (arange, linspace, eye, diag, vander).
 *
 * Provides utility functions for creating arrays with specific patterns:
 *   - arange: Create evenly spaced values within an interval
 *   - linspace: Create fixed-length linearly spaced arrays
 *   - eye: Create identity or unit matrices
 *   - diag: Create diagonal matrices or extract diagonals
 *   - vander: Generate Vandermonde matrices
 *
 * @namespace numbits
 */

#pragma once

#include "ndarray.hpp"
#include <vector>
#include <stdexcept>

namespace numbits {

/**
 * @brief Create a 1D array with evenly spaced values.
 *
 * Equivalent to NumPy's `arange(start, stop, step)`.
 *
 * Values are generated starting at `start`, incremented by `step`, and
 * continuing while:
 *   - `value < stop` for positive steps
 *   - `value > stop` for negative steps
 *
 * @tparam T Numeric value type.
 * @param start Starting value of sequence.
 * @param stop Generate values up to but not including this value.
 * @param step Spacing between values (must not be zero).
 *
 * @return ndarray<T> A 1D array containing the generated sequence.
 *
 * @throws std::runtime_error If step == 0.
 */
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

/**
 * @brief Create a 1D array ranging from 0 to stop (exclusive).
 *
 * Equivalent to NumPy's `arange(stop)`.
 *
 * @tparam T Numeric type.
 * @param stop Upper limit (exclusive).
 * @return ndarray<T> Created sequence.
 */
template<typename T>
ndarray<T> arange(T stop) {
    return arange<T>(T{0}, stop, T{1});
}

/**
 * @brief Create a 1D array ranging from start to stop (exclusive) with step 1.
 *
 * Equivalent to NumPy's `arange(start, stop)`.
 *
 * @tparam T Numeric type.
 * @param start Starting value.
 * @param stop Ending boundary (exclusive).
 * @return ndarray<T> Generated sequence.
 */
template<typename T>
ndarray<T> arange(T start, T stop) {
    return arange<T>(start, stop, T{1});
}

/**
 * @brief Generate linearly spaced values over an interval.
 *
 * Equivalent to NumPy's `linspace(start, stop, num, endpoint)`.
 *
 * Produces `num` evenly spaced samples:
 *   - If `endpoint == true`, the last value equals `stop`.
 *   - If `endpoint == false`, spacing divides interval into `num` segments.
 *
 * @tparam T Numeric type.
 * @param start First value.
 * @param stop Last value (included or excluded depending on `endpoint`).
 * @param num Number of samples to generate.
 * @param endpoint Whether to include `stop` as the last element.
 *
 * @return ndarray<T> A 1D array of evenly spaced values.
 */
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

/**
 * @brief Create an identity matrix or a diagonal matrix with diagonal offset.
 *
 * Equivalent to NumPy's `eye(n, m, k)`.
 *
 * - If `m == 0`, a square matrix of size `n x n` is created.
 * - Ones are placed on the diagonal offset by `k`.
 *   - `k = 0`: main diagonal
 *   - `k > 0`: upper diagonal
 *   - `k < 0`: lower diagonal
 *
 * @tparam T Numeric type.
 * @param n Number of rows.
 * @param m Number of columns (optional, default = n).
 * @param k Diagonal offset.
 *
 * @return ndarray<T> Identity-like matrix.
 */
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

/**
 * @brief Create a diagonal matrix or extract a diagonal from a matrix.
 *
 * Equivalent to NumPy's `diag(arr, k)`.
 *
 * Behavior:
 *   - If input is 1D:
 *        Produces a diagonal matrix with elements of arr placed at diagonal k.
 *   - If input is 2D:
 *        Extracts the diagonal at offset k.
 *
 * @tparam T Numeric type.
 * @param arr Input array (1D or 2D).
 * @param k Diagonal offset (same meaning as in eye).
 *
 * @return ndarray<T> Created diagonal matrix or extracted diagonal.
 *
 * @throws std::runtime_error If input is not 1D or 2D.
 */
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

    // 2D array → extract diagonal
    else if (shape.size() == 2) {
        size_t rows = shape[0];
        size_t cols = shape[1];
        
        int start_row = k < 0 ? -k : 0;
        int start_col = k > 0 ? k : 0;

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

/**
 * @brief Generate a Vandermonde matrix from a 1D array.
 *
 * Equivalent to NumPy's `vander(x, N, increasing)`.
 *
 * Each row i of the output is:
 *   - `[x[i]^(N-1), x[i]^(N-2), ..., x[i]^0]`  (default)
 *   - `[x[i]^0, x[i]^1, ..., x[i]^(N-1)]`      (if increasing == true)
 *
 * @tparam T Numeric type.
 * @param x Input 1D array.
 * @param N Number of columns in output.
 * @param increasing Whether power order is ascending.
 *
 * @return ndarray<T> Vandermonde matrix of shape (M, N),
 *         where M = x.size().
 *
 * @throws std::runtime_error If input is not 1D.
 */
template<typename T>
ndarray<T> vander(const ndarray<T>& x, size_t N, bool increasing = false) {
    const auto& shape = x.shape();

    if (shape.size() != 1) {
        throw std::runtime_error("vander: input must be a 1D ndarray");
    }

    size_t M = shape[0];
    ndarray<T> result({M, N});
    result.fill(T{0});

    for (size_t i = 0; i < M; ++i) {
        T base = x[i];

        for (size_t j = 0; j < N; ++j) {
            size_t power = increasing ? j : (N - 1 - j);

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
