/**
 * @file operations.hpp
 * @brief Element-wise operations, scalar operations, reductions, logical and comparison operations for ndarrays.
 *
 * Provides a comprehensive set of operations for n-dimensional arrays (ndarray), including:
 *  - Element-wise arithmetic (add, subtract, multiply, divide)
 *  - Scalar arithmetic (add_scalar, multiply_scalar, etc.)
 *  - Reduction operations (sum, mean, min, max, all, any)
 *  - Cumulative operations (cumsum, cumprod)
 *  - Comparison operations (equal, not_equal, less, greater, etc.)
 *  - Logical operations (logical_and, logical_or, logical_xor, logical_not)
 *  - Advanced operations (clip, argmax, argmin)
 *  - Operator overloads for intuitive syntax
 *
 * @namespace numbits
 */

#pragma once

#include "ndarray.hpp"
#include "broadcasting.hpp"
#include "utils.hpp"
#include <functional>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace numbits {

/**
 * @brief Element-wise addition of two ndarrays with broadcasting.
 * @tparam T Element type
 * @param a First ndarray
 * @param b Second ndarray
 * @return ndarray containing element-wise sum of a and b
 * @throws std::runtime_error if shapes are incompatible
 * @note Broadcasting is performed similar to NumPy.
 * @complexity O(n), where n is the total number of elements after broadcasting
 */
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

/**
 * @brief Element-wise subtraction of two ndarrays with broadcasting.
 * @tparam T Element type
 * @param a Minuend ndarray
 * @param b Subtrahend ndarray
 * @return ndarray containing element-wise difference (a - b)
 * @throws std::runtime_error if shapes are incompatible
 * @note Broadcasting is performed similar to NumPy.
 * @complexity O(n)
 */
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

/**
 * @brief Element-wise multiplication of two ndarrays with broadcasting.
 * @tparam T Element type
 * @param a First ndarray
 * @param b Second ndarray
 * @return ndarray containing element-wise product of a and b
 * @throws std::runtime_error if shapes are incompatible
 * @note Broadcasting is performed similar to NumPy.
 * @complexity O(n)
 */
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

/**
 * @brief Element-wise division of two ndarrays with broadcasting.
 * @tparam T Element type
 * @param a Dividend ndarray
 * @param b Divisor ndarray
 * @return ndarray containing element-wise quotient (a / b)
 * @throws std::runtime_error if shapes are incompatible
 * @note Broadcasting is performed similar to NumPy.
 * @complexity O(n)
 */
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

// Scalar Operations

/**
 * @brief Adds a scalar to each element of the ndarray.
 * @tparam T Element type
 * @param a Input ndarray
 * @param scalar Scalar to add
 * @return ndarray with each element increased by scalar
 * @complexity O(n)
 */
template<typename T>
ndarray<T> add_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val + scalar; });
    return result;
}

/**
 * @brief Subtracts a scalar from each element of the ndarray.
 * @tparam T Element type
 * @param a Input ndarray
 * @param scalar Scalar to subtract
 * @return ndarray with each element decreased by scalar
 * @complexity O(n)
 */
template<typename T>
ndarray<T> subtract_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val - scalar; });
    return result;
}

/**
 * @brief Multiplies each element of the ndarray by a scalar.
 * @tparam T Element type
 * @param a Input ndarray
 * @param scalar Scalar multiplier
 * @return ndarray with each element multiplied by scalar
 * @complexity O(n)
 */
template<typename T>
ndarray<T> multiply_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val * scalar; });
    return result;
}

/**
 * @brief Divides each element of the ndarray by a scalar.
 * @tparam T Element type
 * @param a Input ndarray
 * @param scalar Scalar divisor
 * @return ndarray with each element divided by scalar
 * @complexity O(n)
 */
template<typename T>
ndarray<T> divide_scalar(const ndarray<T>& a, T scalar) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return val / scalar; });
    return result;
}

// Value Clipping

/**
 * @brief Clips values of an ndarray element-wise between min and max arrays.
 * @tparam T Element type
 * @param arr Input ndarray
 * @param min_vals Minimum values (ndarray)
 * @param max_vals Maximum values (ndarray)
 * @return ndarray with clipped values
 * @throws std::runtime_error if min_vals > max_vals after broadcasting
 * @note Broadcasting is applied to arr, min_vals, max_vals to match shapes
 * @complexity O(n)
 */
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

/**
 * @brief Clips values of an ndarray element-wise between scalar min and max.
 * @tparam T Element type
 * @param arr Input ndarray
 * @param min_value Minimum value
 * @param max_value Maximum value
 * @return ndarray with clipped values
 * @throws std::runtime_error if min_value > max_value
 * @complexity O(n)
 */
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

//  Logical Operations

/**
 * @brief Computes element-wise logical AND of two ndarrays.
 * @tparam T Element type
 * @param a First ndarray
 * @param b Second ndarray
 * @return ndarray<bool> containing element-wise logical AND
 * @note Broadcasting is applied
 * @complexity O(n)
 */
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

/**
 * @brief Computes element-wise logical OR of two ndarrays.
 * @tparam T Element type
 * @param a First ndarray
 * @param b Second ndarray
 * @return ndarray<bool> containing element-wise logical OR
 * @note Broadcasting is applied
 * @complexity O(n)
 */
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

/**
 * @brief Computes element-wise logical XOR of two ndarrays.
 * @tparam T Element type
 * @param a First ndarray
 * @param b Second ndarray
 * @return ndarray<bool> containing element-wise logical XOR
 * @note Broadcasting is applied
 * @complexity O(n)
 */
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

/**
 * @brief Computes element-wise logical NOT of an ndarray.
 * @tparam T Element type
 * @param a Input ndarray
 * @return ndarray<bool> containing logical NOT of each element
 * @complexity O(n)
 */
template<typename T>
ndarray<bool> logical_not(const ndarray<T>& a) {
    ndarray<bool> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [](const T& val) { return !static_cast<bool>(val); });
    return result;
}

// Comparison Operations

// All comparison operations: equal, not_equal, less, greater, less_equal, greater_equal
// Each has broadcasting support and element-wise boolean output

/**
 * @brief Computes element-wise equality of two ndarrays.
 */
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

// Reduction Operations

/**
 * @brief Computes sum of all elements in ndarray.
 */
template<typename T>
T sum(const ndarray<T>& arr) {
    return std::accumulate(arr.begin(), arr.end(), T{0});
}

/**
 * @brief Computes mean of all elements in ndarray.
 */
template<typename T>
T mean(const ndarray<T>& arr) {
    if (arr.size() == 0) return T{0};
    return sum(arr) / static_cast<T>(arr.size());
}

/**
 * @brief Returns minimum element of ndarray.
 * @throws std::runtime_error if ndarray is empty
 */
template<typename T>
T min(const ndarray<T>& arr) {
    if (arr.size() == 0) throw std::runtime_error("Cannot find min of empty ndarray");
    return *std::min_element(arr.begin(), arr.end());
}

/**
 * @brief Returns maximum element of ndarray.
 * @throws std::runtime_error if ndarray is empty
 */
template<typename T>
T max(const ndarray<T>& arr) {
    if (arr.size() == 0) throw std::runtime_error("Cannot find max of empty ndarray");
    return *std::max_element(arr.begin(), arr.end());
}

/**
 * @brief Checks if all elements are true/nonzero.
 */
template<typename T>
bool all(const ndarray<T>& arr) {
    return std::all_of(arr.begin(), arr.end(),
                       [](const T& value) { return static_cast<bool>(value); });
}

/**
 * @brief Checks if any element is true/nonzero.
 */
template<typename T>
bool any(const ndarray<T>& arr) {
    return std::any_of(arr.begin(), arr.end(),
                       [](const T& value) { return static_cast<bool>(value); });
}

/**
 * @brief Computes cumulative sum of ndarray elements.
 */
template<typename T>
ndarray<T> cumsum(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    if (arr.size() == 0) return result;
    std::partial_sum(arr.begin(), arr.end(), result.begin());
    return result;
}

/**
 * @brief Computes cumulative product of ndarray elements.
 */
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

/**
 * @brief Returns index of maximum element.
 * @throws std::runtime_error if ndarray is empty
 */
template<typename T>
size_t argmax(const ndarray<T>& arr) {
    if (arr.size() == 0) throw std::runtime_error("Cannot compute argmax of empty ndarray");
    return static_cast<size_t>(std::distance(arr.begin(), std::max_element(arr.begin(), arr.end())));
}

/**
 * @brief Returns index of minimum element.
 * @throws std::runtime_error if ndarray is empty
 */
template<typename T>
size_t argmin(const ndarray<T>& arr) {
    if (arr.size() == 0) throw std::runtime_error("Cannot compute argmin of empty ndarray");
    return static_cast<size_t>(std::distance(arr.begin(), std::min_element(arr.begin(), arr.end())));
}

// Operator Overloads

// Arithmetic operators +, -, *, / (elementwise and scalar)
// Unary minus operator

template<typename T>
ndarray<T> operator+(const ndarray<T>& a, const ndarray<T>& b) { return add(a, b); }
template<typename T>
ndarray<T> operator-(const ndarray<T>& a, const ndarray<T>& b) { return subtract(a, b); }
template<typename T>
ndarray<T> operator*(const ndarray<T>& a, const ndarray<T>& b) { return multiply(a, b); }
template<typename T>
ndarray<T> operator/(const ndarray<T>& a, const ndarray<T>& b) { return divide(a, b); }

template<typename T>
ndarray<T> operator+(const ndarray<T>& a, T scalar) { return add_scalar(a, scalar); }
template<typename T>
ndarray<T> operator+(T scalar, const ndarray<T>& a) { return add_scalar(a, scalar); }

template<typename T>
ndarray<T> operator-(const ndarray<T>& a, T scalar) { return subtract_scalar(a, scalar); }
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
ndarray<T> operator*(const ndarray<T>& a, T scalar) { return multiply_scalar(a, scalar); }
template<typename T>
ndarray<T> operator*(T scalar, const ndarray<T>& a) { return multiply_scalar(a, scalar); }

template<typename T>
ndarray<T> operator/(const ndarray<T>& a, T scalar) { return divide_scalar(a, scalar); }
template<typename T>
ndarray<T> operator/(T scalar, const ndarray<T>& a) {
    ndarray<T> result(a.shape());
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](T val) { return scalar / val; });
    return result;
}

} // namespace numbits
