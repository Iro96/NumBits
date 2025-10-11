#pragma once
#include "../core/ndarray.hpp"

namespace nb {

template<typename T>
T dot(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.size() != b.size()) throw std::runtime_error("Shape mismatch in dot()");
    T result = 0;
    for (size_t i = 0; i < a.size(); ++i)
        result += a(i) * b(i);
    return result;
}

} // namespace nb
