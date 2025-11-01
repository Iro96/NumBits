#pragma once
#include <vector>
#include <numeric>
#include <functional>
#include <limits>
#include <stdexcept>

namespace numbits {

// runtime only; cannot be constexpr with std::vector
inline size_t total_size(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        if (dim && total > std::numeric_limits<size_t>::max() / dim)
            throw std::overflow_error("total_size(): overflow in shape product");
        total *= dim;
    }
    return total;
}

} // namespace numbits
