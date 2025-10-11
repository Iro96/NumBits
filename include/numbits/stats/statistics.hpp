#pragma once
#include "../core/ndarray.hpp"
#include "../ops/reduction.hpp"
#include <cmath>

namespace numbits {

template <typename T>
T variance(const ndarray<T>& A) {
    T m = mean(A);
    T var = 0;
    for (auto& x : A.data()) var += (x - m) * (x - m);
    return var / static_cast<T>(A.size());
}

template <typename T>
T stddev(const ndarray<T>& A) {
    return std::sqrt(variance(A));
}

} // namespace nb
