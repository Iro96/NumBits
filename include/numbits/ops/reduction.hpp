#pragma once
#include "../core/ndarray.hpp"

namespace numbits {

template <typename T>
T sum(const ndarray<T>& A) {
    T s = 0;
    for (auto& v : A.data()) s += v;
    return s;
}

template <typename T>
T mean(const ndarray<T>& A) {
    return sum(A) / static_cast<T>(A.size());
}

} // namespace nb
