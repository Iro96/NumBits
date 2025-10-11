#pragma once
#include "../core/ndarray.hpp"
#include <random>

namespace nb {

template<typename T = double>
ndarray<T> rand(std::vector<size_t> shape) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    std::vector<T> data(total);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    for (auto& v : data) v = dist(gen);
    ndarray<T> result;
    result = ndarray<T>(shape);
    for (size_t i = 0; i < total; ++i) result.data()[i] = data[i];
    return result;
}

} // namespace nb
