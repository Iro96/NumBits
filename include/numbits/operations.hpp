#pragma once

#include "ndarray.hpp"
#include "broadcasting.hpp"
#include "utils.hpp"
#include <functional>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace numbits {

// Element-wise operations
template<typename T>
ndarray<T> add(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::plus<T>());
    
    return result;
}

template<typename T>
ndarray<T> subtract(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::minus<T>());
    
    return result;
}

template<typename T>
ndarray<T> multiply(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::multiplies<T>());
    
    return result;
}

template<typename T>
ndarray<T> divide(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::divides<T>());
    
    return result;
}

// Scalar operations
template<typename T>
ndarray<T> add_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val + scalar; });
    return result;
}

template<typename T>
ndarray<T> subtract_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val - scalar; });
    return result;
}

template<typename T>
ndarray<T> multiply_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val * scalar; });
    return result;
}

template<typename T>
ndarray<T> divide_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val / scalar; });
    return result;
}

// Value clipping (NumPy-style)
template<typename T>
ndarray<T> clip(const ndarray<T>& arr, const ndarray<T>& min_vals, const ndarray<T>& max_vals) {
    Shape minmax_shape = broadcast_shapes(min_vals.shape(), max_vals.shape());
    Shape target_shape = broadcast_shapes(arr.shape(), minmax_shape);

    ndarray<T> source = broadcast_to(arr, target_shape);
    ndarray<T> min_b = broadcast_to(min_vals, target_shape);
    ndarray<T> max_b = broadcast_to(max_vals, target_shape);

    ndarray<T> result(target_shape);
    for (size_t i = 0; i < result.size(); ++i) {
        if (min_b[i] > max_b[i]) {
            throw std::runtime_error("clip: min value greater than max value after broadcasting");
        }
        result[i] = std::min(std::max(source[i], min_b[i]), max_b[i]);
    }
    return result;
}

template<typename T>
ndarray<T> clip(const ndarray<T>& arr, T min_value, T max_value) {
    if (min_value > max_value) {
        throw std::runtime_error("clip: min value greater than max value");
    }
    ndarray<T> result(arr.shape());
    for (size_t i = 0; i < arr.size(); ++i) {
        result[i] = std::min(std::max(arr[i], min_value), max_value);
    }
    return result;
}

// Logical operations
template<typename T>
ndarray<bool> logical_and(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);

    ndarray<bool> result(result_shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<bool>(a_broadcast[i]) && static_cast<bool>(b_broadcast[i]);
    }
    return result;
}

template<typename T>
ndarray<bool> logical_or(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);

    ndarray<bool> result(result_shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<bool>(a_broadcast[i]) || static_cast<bool>(b_broadcast[i]);
    }
    return result;
}

template<typename T>
ndarray<bool> logical_xor(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);

    ndarray<bool> result(result_shape);
    for (size_t i = 0; i < result.size(); ++i) {
        bool left = static_cast<bool>(a_broadcast[i]);
        bool right = static_cast<bool>(b_broadcast[i]);
        result[i] = left != right;
    }
    return result;
}

template<typename T>
ndarray<bool> logical_not(const ndarray<T>& a) {
    ndarray<bool> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [](const T& val) { return !static_cast<bool>(val); });
    return result;
}

// Operator overloads for convenience
template<typename T>
ndarray<T> operator+(const ndarray<T>& a, const ndarray<T>& b) {
    return add(a, b);
}

template<typename T>
ndarray<T> operator-(const ndarray<T>& a, const ndarray<T>& b) {
    return subtract(a, b);
}

template<typename T>
ndarray<T> operator*(const ndarray<T>& a, const ndarray<T>& b) {
    return multiply(a, b);
}

template<typename T>
ndarray<T> operator/(const ndarray<T>& a, const ndarray<T>& b) {
    return divide(a, b);
}

template<typename T>
ndarray<T> operator+(const ndarray<T>& a, T scalar) {
    return add_scalar(a, scalar);
}

template<typename T>
ndarray<T> operator+(T scalar, const ndarray<T>& a) {
    return add_scalar(a, scalar);
}

template<typename T>
ndarray<T> operator-(const ndarray<T>& a, T scalar) {
    return subtract_scalar(a, scalar);
}

template<typename T>
ndarray<T> operator-(T scalar, const ndarray<T>& a) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return scalar - val; });
    return result;
}

template<typename T>
ndarray<T> operator-(const ndarray<T>& a) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [](T val) { return -val; });
    return result;
}

template<typename T>
ndarray<T> operator*(const ndarray<T>& a, T scalar) {
    return multiply_scalar(a, scalar);
}

template<typename T>
ndarray<T> operator*(T scalar, const ndarray<T>& a) {
    return multiply_scalar(a, scalar);
}

template<typename T>
ndarray<T> operator/(const ndarray<T>& a, T scalar) {
    return divide_scalar(a, scalar);
}

template<typename T>
ndarray<T> operator/(T scalar, const ndarray<T>& a) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return scalar / val; });
    return result;
}

// Comparison operations
template<typename T>
ndarray<bool> equal(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<bool> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::equal_to<T>());
    
    return result;
}

template<typename T>
ndarray<bool> not_equal(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<bool> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::not_equal_to<T>());
    
    return result;
}

template<typename T>
ndarray<bool> less(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<bool> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::less<T>());
    
    return result;
}

template<typename T>
ndarray<bool> greater(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<bool> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::greater<T>());
    
    return result;
}

template<typename T>
ndarray<bool> less_equal(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<bool> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::less_equal<T>());
    
    return result;
}

template<typename T>
ndarray<bool> greater_equal(const ndarray<T>& a, const ndarray<T>& b) {
    Shape result_shape = broadcast_shapes(a.shape(), b.shape());
    ndarray<bool> result(result_shape);
    
    ndarray<T> a_broadcast = broadcast_to(a, result_shape);
    ndarray<T> b_broadcast = broadcast_to(b, result_shape);
    
    std::transform(a_broadcast.begin(), a_broadcast.end(), 
                   b_broadcast.begin(), result.begin(),
                   std::greater_equal<T>());
    
    return result;
}

// Reduction operations
template<typename T>
T sum(const ndarray<T>& arr) {
    return std::accumulate(arr.begin(), arr.end(), T{0});
}

template<typename T>
T mean(const ndarray<T>& arr) {
    if (arr.size() == 0) return T{0};
    return sum(arr) / static_cast<T>(arr.size());
}

template<typename T>
T min(const ndarray<T>& arr) {
    if (arr.size() == 0) throw std::runtime_error("Cannot find min of empty ndarray");
    return *std::min_element(arr.begin(), arr.end());
}

template<typename T>
T max(const ndarray<T>& arr) {
    if (arr.size() == 0) throw std::runtime_error("Cannot find max of empty ndarray");
    return *std::max_element(arr.begin(), arr.end());
}

template<typename T>
bool all(const ndarray<T>& arr) {
    return std::all_of(arr.begin(), arr.end(),
                       [](const T& value) { return static_cast<bool>(value); });
}

template<typename T>
bool any(const ndarray<T>& arr) {
    return std::any_of(arr.begin(), arr.end(),
                       [](const T& value) { return static_cast<bool>(value); });
}

template<typename T>
ndarray<T> cumsum(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    if (arr.size() == 0) {
        return result;
    }
    std::partial_sum(arr.begin(), arr.end(), result.begin());
    return result;
}

template<typename T>
ndarray<T> cumprod(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    T running = T{1};
    for (size_t i = 0; i < arr.size(); ++i) {
        running *= arr[i];
        result[i] = running;
    }
    return result;
}

template<typename T>
size_t argmax(const ndarray<T>& arr) {
    if (arr.size() == 0) {
        throw std::runtime_error("Cannot compute argmax of empty ndarray");
    }
    return static_cast<size_t>(std::distance(arr.begin(), std::max_element(arr.begin(), arr.end())));
}

template<typename T>
size_t argmin(const ndarray<T>& arr) {
    if (arr.size() == 0) {
        throw std::runtime_error("Cannot compute argmin of empty ndarray");
    }
    return static_cast<size_t>(std::distance(arr.begin(), std::min_element(arr.begin(), arr.end())));
}

} // namespace numbits
