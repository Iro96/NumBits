#pragma once
#include "../core/ndarray.hpp"
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <functional>

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
 *        Always returns double for precision safety.
 * @throws std::domain_error if A.size() == 0
 */
template <typename T>
double mean(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "numbits::mean requires arithmetic T");

    if (A.size() == 0)
        throw std::domain_error("mean: cannot compute mean of empty ndarray");

    return static_cast<double>(sum(A)) /
           static_cast<double>(A.size());
}

/**
 * @brief Compute mean of all elements in an ndarray, truncating to integer result.
 *
 * Performs integer division (truncates fractional part).
 * Uses a widened accumulator to prevent overflow, and checks
 * that the result fits into T before narrowing.
 *
 * @tparam T Integral type.
 * @throws std::domain_error if A.size() == 0
 * @throws std::overflow_error if the mean cannot fit into T
 */
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
mean_truncated(const ndarray<T>& A) {
    if (A.size() == 0)
        throw std::domain_error("mean_truncated: cannot compute mean of empty ndarray");

    using Wide = long long;  // wide enough for most integer accumulation
    static_assert(std::numeric_limits<Wide>::digits >= std::numeric_limits<T>::digits,
                  "mean_truncated: accumulator type too small");

    Wide total = 0;
    for (const auto& v : A.data())
        total += static_cast<Wide>(v);

    Wide divisor = static_cast<Wide>(A.size());
    Wide quotient = total / divisor;

    if (quotient > static_cast<Wide>(std::numeric_limits<T>::max()) ||
        quotient < static_cast<Wide>(std::numeric_limits<T>::min())) {
        throw std::overflow_error("mean_truncated: result out of range for target type");
    }

    return static_cast<T>(quotient);
}

/**
 * @brief Helper function to compute reduction along a specific axis.
 */
template <typename T, typename BinaryOp, typename InitOp>
ndarray<T> reduce_axis(const ndarray<T>& A, int axis, BinaryOp op, InitOp init) {
    if (A.size() == 0) {
        throw std::domain_error("reduce_axis: cannot reduce empty ndarray");
    }

    if (axis == -1) {
        // Reduce all elements to scalar
        T result = init();
        for (const auto& v : A.data()) {
            result = op(result, v);
        }
        return ndarray<T>({1}, result);  // Return 0-D array
    }

    size_t ax = static_cast<size_t>(axis);
    if (ax >= A.shape().size()) {
        throw std::invalid_argument("reduce_axis: axis out of bounds");
    }

    std::vector<size_t> new_shape = A.shape();
    new_shape.erase(new_shape.begin() + ax);

    if (new_shape.empty()) {
        // Result is scalar
        T result = init();
        for (const auto& v : A.data()) {
            result = op(result, v);
        }
        return ndarray<T>({1}, result);
    }

    ndarray<T> result(new_shape, T{});

    // Compute strides for iteration
    std::vector<size_t> outer_strides = *A.strides();
    outer_strides.erase(outer_strides.begin() + ax);

    size_t axis_size = A.shape()[ax];
    size_t axis_stride = (*A.strides())[ax];

    // Iterate over all elements in the result shape
    std::vector<size_t> indices(new_shape.size(), 0);
    bool done = false;
    while (!done) {
        // Compute flat index for result
        size_t result_flat_idx = 0;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            result_flat_idx += indices[i] * outer_strides[i];
        }

        // Reduce along the axis
        T val = init();
        for (size_t i = 0; i < axis_size; ++i) {
            size_t data_idx = result_flat_idx + i * axis_stride;
            val = op(val, A.data()[data_idx]);
        }
        result[result_flat_idx] = val;

        // Increment indices
        size_t i = 0;
        while (i < indices.size()) {
            if (++indices[i] < new_shape[i]) break;
            indices[i] = 0;
            ++i;
        }
        if (i == indices.size()) done = true;
    }

    return result;
}

/**
 * @brief Compute sum along specified axis.
 * @param A Input ndarray
 * @param axis Axis to reduce along, or -1 to reduce all
 * @return ndarray<T> with reduced shape
 */
template <typename T>
ndarray<T> sum_axis(const ndarray<T>& A, int axis = -1) {
    static_assert(std::is_arithmetic_v<T>, "numbits::sum requires arithmetic T");
    return reduce_axis(A, axis, std::plus<T>{}, []() { return T(0); });
}

/**
 * @brief Compute mean along specified axis.
 * @param A Input ndarray
 * @param axis Axis to reduce along, or -1 to reduce all
 * @return ndarray<double> with reduced shape (always double for precision)
 */
template <typename T>
ndarray<double> mean_axis(const ndarray<T>& A, int axis = -1) {
    static_assert(std::is_arithmetic_v<T>, "numbits::mean requires arithmetic T");
    if (A.size() == 0) {
        throw std::domain_error("mean: cannot compute mean of empty ndarray");
    }

    if (axis == -1) {
        return ndarray<double>({1}, static_cast<double>(sum(A).data()[0]) / A.size());
    }

    size_t ax = static_cast<size_t>(axis);
    if (ax >= A.shape().size()) {
        throw std::invalid_argument("mean: axis out of bounds");
    }

    ndarray<T> sum_result = sum_axis(A, axis);
    size_t axis_size = A.shape()[ax];
    ndarray<double> result(sum_result.shape());

    for (size_t i = 0; i < sum_result.size(); ++i) {
        result[i] = static_cast<double>(sum_result[i]) / axis_size;
    }

    return result;
}

/**
 * @brief Compute maximum along specified axis.
 * @param A Input ndarray
 * @param axis Axis to reduce along, or -1 to reduce all
 * @return ndarray<T> with reduced shape
 */
template <typename T>
ndarray<T> max_axis(const ndarray<T>& A, int axis = -1) {
    static_assert(std::is_arithmetic_v<T>, "numbits::max requires arithmetic T");
    return reduce_axis(A, axis, [](T a, T b) { return std::max(a, b); }, []() { return std::numeric_limits<T>::lowest(); });
}

/**
 * @brief Compute minimum along specified axis.
 * @param A Input ndarray
 * @param axis Axis to reduce along, or -1 to reduce all
 * @return ndarray<T> with reduced shape
 */
template <typename T>
ndarray<T> min_axis(const ndarray<T>& A, int axis = -1) {
    static_assert(std::is_arithmetic_v<T>, "numbits::min requires arithmetic T");
    return reduce_axis(A, axis, [](T a, T b) { return std::min(a, b); }, []() { return std::numeric_limits<T>::max(); });
}

} // namespace numbits
