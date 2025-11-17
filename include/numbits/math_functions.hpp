#pragma once

#include "ndarray.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace numbits {

// Mathematical functions
template<typename T>
ndarray<T> abs(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::abs(val); });
    return result;
}

template<typename T>
ndarray<T> sqrt(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::sqrt(val); });
    return result;
}

template<typename T>
ndarray<T> pow(const ndarray<T>& arr, T exponent) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [exponent](T val) { return std::pow(val, exponent); });
    return result;
}

template<typename T>
ndarray<T> exp(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::exp(val); });
    return result;
}

template<typename T>
ndarray<T> log(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::log(val); });
    return result;
}

template<typename T>
ndarray<T> log10(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::log10(val); });
    return result;
}

template<typename T>
ndarray<T> sin(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::sin(val); });
    return result;
}

template<typename T>
ndarray<T> cos(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::cos(val); });
    return result;
}

template<typename T>
ndarray<T> tan(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::tan(val); });
    return result;
}

template<typename T>
ndarray<T> asin(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::asin(val); });
    return result;
}

template<typename T>
ndarray<T> acos(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::acos(val); });
    return result;
}

template<typename T>
ndarray<T> atan(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::atan(val); });
    return result;
}

template<typename T>
ndarray<T> sinh(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::sinh(val); });
    return result;
}

template<typename T>
ndarray<T> cosh(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::cosh(val); });
    return result;
}

template<typename T>
ndarray<T> tanh(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::tanh(val); });
    return result;
}

template<typename T>
ndarray<T> ceil(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::ceil(val); });
    return result;
}

template<typename T>
ndarray<T> floor(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::floor(val); });
    return result;
}

template<typename T>
ndarray<T> round(const ndarray<T>& arr) {
    ndarray<T> result(arr.shape());
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [](T val) { return std::round(val); });
    return result;
}

} // namespace numbits

