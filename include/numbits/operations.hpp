#pragma once

#include "ndarray.hpp"
#include "broadcasting.hpp"
#include "utils.hpp"
#include <functional>
#include <algorithm>
#include <numeric>

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

} // namespace numbits

