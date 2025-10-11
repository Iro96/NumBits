#pragma once
#include <vector>
#include <numeric>
#include <algorithm>

namespace numbits {

template <typename T>
T sum(const std::vector<T>& data) {
    return std::accumulate(data.begin(), data.end(), T(0));
}

template <typename T>
T mean(const std::vector<T>& data) {
    if (data.empty()) {
        return T(0);
    }
    return sum(data) / static_cast<T>(data.size());
}

} // namespace nb
