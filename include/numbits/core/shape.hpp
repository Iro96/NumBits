#pragma once
#include <vector>
#include <numeric>

namespace numbits {

inline size_t total_size(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
}

} // namespace nb
