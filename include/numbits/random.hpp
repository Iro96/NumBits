#pragma once

#include "ndarray.hpp"
#include <random>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace numbits{

/**
 * @brief Returns the default 32-bit random engine.
 * 
 * This engine is shared across all random functions in this namespace.
 * It is initialized with a non-deterministic seed from std::random_device.
 * 
 * @return std::mt19937& Reference to the default engine.
 */
inline std::mt19937& default_engine() {
    static std::mt19937 eng(std::random_device{}());
    return eng;
}

/**
 * @brief Returns the default 64-bit random engine.
 * 
 * This engine can be used when higher-quality 64-bit random numbers are needed.
 * Initialized with a non-deterministic seed from std::random_device.
 * 
 * @return std::mt19937_64& Reference to the 64-bit default engine.
 */
inline std::mt19937_64& default_engine64() {
    static std::mt19937_64 eng(std::random_device{}());
    return eng;
}

/**
 * @brief Seeds all default engines for reproducible random numbers.
 * 
 * @param seed Unsigned long seed value to initialize both default engines.
 */
inline void seed_engine(unsigned long seed) {
    default_engine().seed(seed);
    default_engine64().seed(seed);
}

/**
 * @brief Fills an ndarray with random numbers from a given distribution.
 * 
 * This function can optionally fill the array in parallel using OpenMP
 * if `parallel` is true and the array is large.
 * 
 * @tparam T Type of the elements in the ndarray.
 * @tparam Dist Type of the random distribution (e.g., std::uniform_real_distribution).
 * @tparam Engine Type of the random engine.
 * @param arr Reference to the ndarray to fill.
 * @param dist Random distribution to generate numbers from.
 * @param eng Random engine to use for generating numbers.
 * @param parallel Whether to use parallel execution (default: false).
 */
template<typename T, typename Dist, typename Engine>
void fill_ndarray(ndarray<T>& arr, Dist& dist, Engine& eng, bool parallel = false) {
    if(parallel && arr.size() > 1000) { // arbitrary threshold
#ifdef _OPENMP
        #pragma omp parallel for
        for(size_t i = 0; i < arr.size(); ++i) {
            // Each thread uses the same engine (not ideal for perfect reproducibility)
            arr[i] = dist(eng);
        }
#else
        for(size_t i = 0; i < arr.size(); ++i)
            arr[i] = dist(eng);
#endif
    } else {
        for(size_t i = 0; i < arr.size(); ++i)
            arr[i] = dist(eng);
    }
}

/**
 * @brief Generates an ndarray filled with random numbers from a uniform distribution.
 * 
 * @tparam T Floating-point type (default: float)
 * @tparam Engine Random engine type (default: std::mt19937)
 * @param shape Shape of the ndarray to generate.
 * @param min_val Minimum value of the distribution (inclusive, default: 0).
 * @param max_val Maximum value of the distribution (inclusive, default: 1).
 * @param eng Random engine to use (default: default_engine()).
 * @param parallel Whether to fill the array in parallel using OpenMP (default: false).
 * @return ndarray<T> Filled ndarray with random numbers.
 */
template<typename T=float, typename Engine = std::mt19937>
ndarray<T> uniform(const Shape& shape, T min_val = T{0}, T max_val = T{1}, 
                   Engine& eng = default_engine(), bool parallel = false) {
    ndarray<T> arr(shape);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    fill_ndarray(arr, dist, eng, parallel);
    return arr;
}

/**
 * @brief Generates an ndarray filled with random numbers from a normal distribution.
 * 
 * @tparam T Floating-point type (default: float)
 * @tparam Engine Random engine type (default: std::mt19937)
 * @param shape Shape of the ndarray to generate.
 * @param mean Mean of the normal distribution (default: 0).
 * @param stddev Standard deviation of the normal distribution (default: 1).
 * @param eng Random engine to use (default: default_engine()).
 * @param parallel Whether to fill the array in parallel using OpenMP (default: false).
 * @return ndarray<T> Filled ndarray with random numbers.
 */
template<typename T=float, typename Engine = std::mt19937>
ndarray<T> normal(const Shape& shape, T mean = T{0}, T stddev = T{1}, 
                  Engine& eng = default_engine(), bool parallel = false) {
    ndarray<T> arr(shape);
    std::normal_distribution<T> dist(mean, stddev);
    fill_ndarray(arr, dist, eng, parallel);
    return arr;
}

/**
 * @brief Generates an ndarray filled with random integers from a uniform integer distribution.
 * 
 * @tparam T Integer type (default: int)
 * @tparam Engine Random engine type (default: std::mt19937)
 * @param shape Shape of the ndarray to generate.
 * @param min_val Minimum integer value (inclusive, default: std::numeric_limits<T>::min()).
 * @param max_val Maximum integer value (inclusive, default: std::numeric_limits<T>::max()).
 * @param eng Random engine to use (default: default_engine()).
 * @param parallel Whether to fill the array in parallel using OpenMP (default: false).
 * @return ndarray<T> Filled ndarray with random integers.
 */
template<typename T=int, typename Engine = std::mt19937>
ndarray<T> randint(const Shape& shape, T min_val = std::numeric_limits<T>::min(),
                   T max_val = std::numeric_limits<T>::max(), 
                   Engine& eng = default_engine(), bool parallel = false) {
    ndarray<T> arr(shape);
    std::uniform_int_distribution<T> dist(min_val, max_val);
    fill_ndarray(arr, dist, eng, parallel);
    return arr;
}

} // namespace numbits::random
