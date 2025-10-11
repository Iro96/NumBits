#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>
#include <type_traits>

namespace numbits {

/**
 * @brief Compute sum of all elements in an ndarray.
 */
template <typename T>
T sum(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "numbits::sum requires arithmetic T");

    T s = T(0);
    for (const auto& v : A.data())
        s += v;

    return s;
}

/**
 * @brief Compute mean of all elements in an ndarray.
 * @throws std::domain_error if A.size() == 0
 */
template <typename T>
T mean(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "numbits::mean requires arithmetic T");

    if (A.size() == 0)
        throw std::domain_error("mean: cannot compute mean of empty ndarray");

    return sum(A) / static_cast<T>(A.size());
}

} // namespace numbits
