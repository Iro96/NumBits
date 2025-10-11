#pragma once
#include "../core/ndarray.hpp"
#include <cmath>

namespace nb {

template<typename T>
ndarray<T> sqrt(const ndarray<T>& a) {
    ndarray<T> out(a.shape());
    for (size_t i = 0; i < a.size(); ++i)
        out(i) = std::sqrt(a(i));
    return out;
}

template<typename T>
ndarray<T> exp(const ndarray<T>& a) {
    ndarray<T> out(a.shape());
    for (size_t i = 0; i < a.size(); ++i)
        out(i) = std::exp(a(i));
    return out;
}

} // namespace nb
