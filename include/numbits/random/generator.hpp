#pragma once
#include "../core/ndarray.hpp"
#include <random>
#include <type_traits>

namespace numbits {

template <typename T = double>
ndarray<T> rand(const std::initializer_list<size_t>& shape) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::rand requires floating-point T (e.g., float or double)");

    ndarray<T> A(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    for (auto& v : A.data())
        v = dist(gen);

    return A;
}

} // namespace numbits
