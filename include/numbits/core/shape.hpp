#pragma once
#pragma once
#include <vector>
#include <numeric>
#include <functional>

namespace numbits {

inline size_t total_size(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
}

} // namespace numbits
