#pragma once

#include "ndarray.hpp"
#include "operations.hpp"
#include <stdexcept>

namespace numbits {

// Matrix multiplication
template<typename T>
ndarray<T> matmul(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("matmul requires 2D ndarrays");
    }
    
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    size_t m = a.shape()[0];
    size_t n = a.shape()[1];
    size_t p = b.shape()[1];
    
    ndarray<T> result(Shape{m, p});
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T sum = T{0};
            for (size_t k = 0; k < n; ++k) {
                sum += a.at({i, k}) * b.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }
    
    return result;
}

// Dot product (generalized)
template<typename T>
ndarray<T> dot(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.ndim() == 1 && b.ndim() == 1) {
        // Vector dot product
        if (a.size() != b.size()) {
            throw std::runtime_error("Vectors must have same size for dot product");
        }
        T result = T{0};
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return ndarray<T>({1}, {result});
    } else if (a.ndim() == 2 && b.ndim() == 2) {
        // Matrix multiplication
        return matmul(a, b);
    } else if (a.ndim() == 2 && b.ndim() == 1) {
        // Matrix-vector multiplication
        if (a.shape()[1] != b.size()) {
            throw std::runtime_error("Matrix and vector dimensions incompatible");
        }
        ndarray<T> result({a.shape()[0]});
        for (size_t i = 0; i < a.shape()[0]; ++i) {
            T sum = T{0};
            for (size_t j = 0; j < a.shape()[1]; ++j) {
                sum += a.at({i, j}) * b[j];
            }
            result[i] = sum;
        }
        return result;
    } else {
        throw std::runtime_error("Unsupported dimensions for dot product");
    }
}

// Transpose
template<typename T>
ndarray<T> transpose(const ndarray<T>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("transpose currently only supports 2D ndarrays");
    }
    
    size_t m = arr.shape()[0];
    size_t n = arr.shape()[1];
    Shape shape = {n, m};
    ndarray<T> result(shape);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result.at({j, i}) = arr.at({i, j});
        }
    }
    
    return result;
}

// Determinant (for 2x2 matrices)
template<typename T>
T determinant(const ndarray<T>& arr) {
    if (arr.ndim() != 2 || arr.shape()[0] != arr.shape()[1]) {
        throw std::runtime_error("determinant requires a square matrix");
    }
    
    if (arr.shape()[0] == 1) {
        return arr.at({0, 0});
    } else if (arr.shape()[0] == 2) {
        return arr.at({0, 0}) * arr.at({1, 1}) - arr.at({0, 1}) * arr.at({1, 0});
    } else {
        // For larger matrices, use recursive expansion (Laplace expansion)
        T det = T{0};
        size_t n = arr.shape()[0];
        
        for (size_t j = 0; j < n; ++j) {
            // Create submatrix
            ndarray<T> submatrix({n - 1, n - 1});
            for (size_t i = 1; i < n; ++i) {
                size_t col_idx = 0;
                for (size_t k = 0; k < n; ++k) {
                    if (k != j) {
                        submatrix.at({i - 1, col_idx}) = arr.at({i, k});
                        col_idx++;
                    }
                }
            }
            
            T sign = (j % 2 == 0) ? T{1} : T{-1};
            det += sign * arr.at({0, j}) * determinant(submatrix);
        }
        
        return det;
    }
}

// Inverse (for square matrices)
template<typename T>
ndarray<T> inverse(const ndarray<T>& arr) {
    if (arr.ndim() != 2 || arr.shape()[0] != arr.shape()[1]) {
        throw std::runtime_error("inverse requires a square matrix");
    }
    
    T det = determinant(arr);
    if (std::abs(det) < 1e-10) {
        throw std::runtime_error("Matrix is singular (determinant is zero)");
    }
    
    size_t n = arr.shape()[0];
    
    if (n == 1) {
        return ndarray<T>({1, 1}, {T{1} / arr.at({0, 0})});
    } else if (n == 2) {
        ndarray<T> result({2, 2});
        T inv_det = T{1} / det;
        result.at({0, 0}) = arr.at({1, 1}) * inv_det;
        result.at({0, 1}) = -arr.at({0, 1}) * inv_det;
        result.at({1, 0}) = -arr.at({1, 0}) * inv_det;
        result.at({1, 1}) = arr.at({0, 0}) * inv_det;
        return result;
    } else {
        // For larger matrices, use adjugate method
        ndarray<T> adjugate({n, n});
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                // Create cofactor matrix
                ndarray<T> cofactor({n - 1, n - 1});
                size_t row_idx = 0;
                for (size_t ii = 0; ii < n; ++ii) {
                    if (ii == i) continue;
                    size_t col_idx = 0;
                    for (size_t jj = 0; jj < n; ++jj) {
                        if (jj == j) continue;
                        cofactor.at({row_idx, col_idx}) = arr.at({ii, jj});
                        col_idx++;
                    }
                    row_idx++;
                }
                
                T sign = ((i + j) % 2 == 0) ? T{1} : T{-1};
                adjugate.at({j, i}) = sign * determinant(cofactor);
            }
        }
        
        return multiply_scalar(adjugate, T{1} / det);
    }
}

// Trace
template<typename T>
T trace(const ndarray<T>& arr) {
    if (arr.ndim() != 2 || arr.shape()[0] != arr.shape()[1]) {
        throw std::runtime_error("trace requires a square matrix");
    }
    
    T result = T{0};
    for (size_t i = 0; i < arr.shape()[0]; ++i) {
        result += arr.at({i, i});
    }
    return result;
}

} // namespace numbits

