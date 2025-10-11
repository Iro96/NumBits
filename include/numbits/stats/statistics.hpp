#pragma once
#include "../ops/reduction.hpp"
#include <cmath>

namespace nb {

template<typename T>
double stddev(const ndarray<T>& a) {
    double m = mean(a);
    double accum = 0.0;
    for (auto& v : a.data()) accum += (v - m) * (v - m);
    return std::sqrt(accum / a.size());
}

} // namespace nb
