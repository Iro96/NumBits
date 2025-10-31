#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>

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
inline double mean(const std::vector<T>& data) noexcept {
    const size_t n = data.size();
    if (n == 0) return 0.0;
    return static_cast<double>(sum(data)) / static_cast<double>(n);
}

} // namespace numbits
