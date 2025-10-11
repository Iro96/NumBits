#pragma once
#include "../core/ndarray.hpp"
#include <numeric>

namespace nb {

template<typename T>
T sum(const ndarray<T>& a) {
    return std::accumulate(a.data().begin(), a.data().end(), T(0));
}

template<typename T>
double mean(const ndarray<T>& a) {
    return static_cast<double>(sum(a)) / static_cast<double>(a.size());
}

} // namespace nb
