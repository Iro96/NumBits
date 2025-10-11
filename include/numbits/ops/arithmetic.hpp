#pragma once
#include "../core/ndarray.hpp"

namespace nb {

template<typename T>
ndarray<T> add(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.size() != b.size()) throw std::runtime_error("Shape mismatch in add()");
    ndarray<T> out(a.shape());
    for (size_t i = 0; i < a.size(); ++i)
        out(i) = a(i) + b(i);
    return out;
}

template<typename T>
ndarray<T> multiply(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.size() != b.size()) throw std::runtime_error("Shape mismatch in multiply()");
    ndarray<T> out(a.shape());
    for (size_t i = 0; i < a.size(); ++i)
        out(i) = a(i) * b(i);
    return out;
}

} // namespace nb
