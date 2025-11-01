#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

namespace numbits {

template <typename T>
inline auto sum(const std::vector<T>& data) noexcept {
    static_assert(std::is_arithmetic_v<T>, "sum() requires arithmetic type");
    using AccT = std::conditional_t<std::is_integral_v<T>, long long, double>;
    AccT total = 0;
    const T* ptr = data.data();
    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) total += ptr[i];
    return total;
}

template <typename T>
double mean(const std::vector<T>& data) {
    if (data.empty())
        throw std::domain_error("mean: cannot compute mean of empty vector");
    return static_cast<double>(sum(data)) / static_cast<double>(data.size());
}

} // namespace numbits
