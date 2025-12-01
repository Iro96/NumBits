#pragma once

#include "ndarray.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace numbits {

// Mathematical functions

/**
 * @brief Element-wise absolute value.
 */
template<typename T>
ndarray<T> abs(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::abs(val); });
    return result;
}

/**
 * @brief Element-wise sign function.
 * Returns -1 for negative, 0 for zero, 1 for positive values.
 */
template<typename T>
ndarray<T> sign(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) -> T { return (val > 0) - (val < 0); });
    return result;
}

/**
 * @brief Element-wise remainder of division (like std::remainder).
 * Arrays a and b must have the same size.
 */
template<typename T>
ndarray<T> remainder(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.size() != b.size())
        throw std::runtime_error("remainder: arrays must have the same size");
    ndarray<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::remainder(a[i], b[i]);
    }
    return result;
}

/**
 * @brief Element-wise clipping of array values to [min_val, max_val].
 */
template<typename T>
ndarray<T> mclip(const ndarray<T>& arr, T min_val, T max_val) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [min_val, max_val](T val) -> T {
                       if (val < min_val) return min_val;
                       if (val > max_val) return max_val;
                       return val;
                   });
    return result;
}

/**
 * @brief 1D linear interpolation.
 * For each x[i], computes corresponding interpolated value using xp and fp arrays.
 * xp must be sorted in ascending order.
 */
template<typename T>
ndarray<T> interp(const ndarray<T>& x, const ndarray<T>& xp, const ndarray<T>& fp) {
    if (xp.size() != fp.size())
        throw std::runtime_error("interp: xp and fp must have the same size");
    if (xp.size() < 2)
        throw std::runtime_error("interp: xp and fp must contain at least 2 points");

    ndarray<T> result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        T xi = x[i];

        // Handle out-of-bounds
        if (xi <= xp[0]) {
            result[i] = fp[0];
            continue;
        }
        if (xi >= xp[xp.size() - 1]) {
            result[i] = fp[fp.size() - 1];
            continue;
        }

        // Find interval
        size_t j = 0;
        while (xi > xp[j + 1]) ++j;

        // Linear interpolation
        T x0 = xp[j], x1 = xp[j + 1];
        T y0 = fp[j], y1 = fp[j + 1];
        result[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0);
    }
    return result;
}

/**
 * @brief Element-wise square root.
 */
template<typename T>
ndarray<T> sqrt(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::sqrt(val); });
    return result;
}

/**
 * @brief Computes element-wise cube root.
 */
template<typename T>
ndarray<T> cbrt(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::cbrt(val); });
    return result;
}

/**
 * @brief Element-wise power function.
 */
template<typename T>
ndarray<T> pow(const ndarray<T>& arr, T exponent) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [exponent](T val) { return std::pow(val, exponent); });
    return result;
}

/**
 * @brief Element-wise exponential function.
 */
template<typename T>
ndarray<T> exp(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::exp(val); });
    return result;
}

/**
 * @brief Computes element-wise exp(x) - 1 accurately for small x.
 *
 * Each element x of the input array is transformed as exp(x) - 1.
 * Using std::expm1 ensures better numerical precision for values of x close to zero.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param arr Input array.
 * @return Array with exp(x) - 1 for each element.
 */
template<typename T>
ndarray<T> expm1(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::expm1(val); });
    return result;
}

/**
 * @brief Element-wise natural logarithm.
 */
template<typename T>
ndarray<T> log(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::log(val); });
    return result;
}

/**
 * @brief Element-wise base-10 logarithm.
 */
template<typename T>
ndarray<T> log10(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::log10(val); });
    return result;
}

/**
 * @brief Element-wise natural logarithm of 1 + x.
 * Computes log(1 + x) accurately, especially for small x.
 */
template<typename T>
ndarray<T> log1p(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::log1p(val); });
    return result;
}

/**
 * @brief Element-wise sine function.
 */
template<typename T>
ndarray<T> sin(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::sin(val); });
    return result;
}

/**
 * @brief Element-wise cosine function.
 */
template<typename T>
ndarray<T> cos(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::cos(val); });
    return result;
}

/**
 * @brief Element-wise tangent function.
 */
template<typename T>
ndarray<T> tan(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::tan(val); });
    return result;
}

/**
 * @brief Element-wise arcsine function.
 */
template<typename T>
ndarray<T> asin(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::asin(val); });
    return result;
}

/**
 * @brief Element-wise arccosine function.
 */
template<typename T>
ndarray<T> acos(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::acos(val); });
    return result;
}

/**
 * @brief Element-wise arctangent function.
 */
template<typename T>
ndarray<T> atan(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::atan(val); });
    return result;
}

/**
 * @brief Element-wise hyperbolic sine function.
 */
template<typename T>
ndarray<T> sinh(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::sinh(val); });
    return result;
}

/**
 * @brief Element-wise hyperbolic cosine function.
 */
template<typename T>
ndarray<T> cosh(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::cosh(val); });
    return result;
}

/**
 * @brief Element-wise hyperbolic tangent function.
 */
template<typename T>
ndarray<T> tanh(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::tanh(val); });
    return result;
}

/**
 * @brief Element-wise ceiling function.
 */
template<typename T>
ndarray<T> ceil(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::ceil(val); });
    return result;
}

/**
 * @brief Element-wise floor function.
 */
template<typename T>
ndarray<T> floor(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::floor(val); });
    return result;
}

/**
 * @brief Element-wise rounding to nearest integer.
 */
template<typename T>
ndarray<T> round(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::round(val); });
    return result;
}

/**
 * @brief Returns element-wise true if the element is NaN.
 */
template<typename T>
ndarray<bool> isnan(const ndarray<T>& arr) {
    ndarray<bool> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::isnan(val); });
    return result;
}

/**
 * @brief Returns element-wise true if the element is infinite (+inf or -inf).
 */
template<typename T>
ndarray<bool> isinf(const ndarray<T>& arr) {
    ndarray<bool> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::isinf(val); });
    return result;
}

} // namespace numbits
