#pragma once
#include "../core/ndarray.hpp"
#include <random>
#include <type_traits>

namespace numbits {

/**
 * @brief Generate a random ndarray with values uniformly distributed in [0, 1).
 *
 * @tparam T Floating-point type (float, double, long double).
 * @param shape Shape of the output array (must not be empty).
 * @return ndarray<T> filled with uniform random numbers.
 */
template <typename T = double>
ndarray<T> rand(const std::initializer_list<size_t>& shape) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::rand requires floating-point T (e.g., float, double)");

    if (shape.size() == 0)
        throw std::invalid_argument("rand: shape cannot be empty");

    ndarray<T> A(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));

    for (auto& v : A.data())
        v = dist(gen);

    return A;
}

/**
 * @brief Generate a random ndarray with values from a normal (Gaussian) distribution.
 *
 * @tparam T Floating-point type.
 * @param shape Shape of the output array.
 * @param mean Mean of the distribution.
 * @param stddev Standard deviation of the distribution.
 * @return ndarray<T> filled with Gaussian random numbers.
 */
template <typename T = double>
ndarray<T> randn(const std::initializer_list<size_t>& shape,
                 T mean = static_cast<T>(0.0),
                 T stddev = static_cast<T>(1.0)) {
    static_assert(std::is_floating_point_v<T>,
                  "numbits::randn requires floating-point T (e.g., float, double)");

    if (shape.size() == 0)
        throw std::invalid_argument("randn: shape cannot be empty");

    ndarray<T> A(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(mean, stddev);

    for (auto& v : A.data())
        v = dist(gen);

    return A;
}

} // namespace numbits
