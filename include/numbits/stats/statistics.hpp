#pragma once
#include "../core/ndarray.hpp"
#include "../ops/reduction.hpp"  // for mean()
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <limits>

namespace numbits {

/**
 * @brief Validate that all elements of an ndarray are finite numbers.
 *
 * Throws std::domain_error if any element is NaN or Inf.
 *
 * @tparam T Numeric type of ndarray
 * @param A Input ndarray
 */
template <typename T>
inline void check_valid_numbers(const ndarray<T>& A) {
    for (const auto& v : A.data()) {
        if (!std::isfinite(static_cast<double>(v))) {
            throw std::domain_error("ndarray contains NaN or Inf");
        }
    }
}

/**
 * @brief Compute the population variance of an ndarray.
 *
 * Computes σ² = (1/N) ∑(x_i - mean)².
 * Supports 1D or 2D arrays.
 *
 * @tparam T Arithmetic type (int, float, double)
 * @param A Input ndarray
 * @return double Population variance of elements
 * @throws std::invalid_argument if A is empty
 * @throws std::domain_error if A contains NaN or Inf
 */
template <typename T>
double variance(const ndarray<T>& A) {
    static_assert(std::is_arithmetic_v<T>, "variance requires numeric T");
    if (A.size() == 0)
        throw std::invalid_argument("variance: empty array");
    check_valid_numbers(A);

    double m = static_cast<double>(mean(A));
    double var = 0.0;
    for (const auto& x : A.data()) {
        double d = static_cast<double>(x) - m;
        var += d * d;
    }
    return var / static_cast<double>(A.size());
}

/**
 * @brief Compute the population standard deviation of an ndarray.
 *
 * Returns σ = sqrt(variance(A)).
 *
 * @tparam T Arithmetic type
 * @param A Input ndarray
 * @return double Standard deviation
 * @throws std::invalid_argument if A is empty
 * @throws std::domain_error if A contains NaN or Inf
 */
template <typename T>
double stddev(const ndarray<T>& A) {
    return std::sqrt(variance<T>(A));
}

/**
 * @brief Compute covariance between two 1D ndarrays.
 *
 * Cov(X, Y) = (1/N) ∑ (X_i - mean(X)) * (Y_i - mean(Y))
 *
 * @tparam T Arithmetic type
 * @param X First 1D ndarray
 * @param Y Second 1D ndarray (same length as X)
 * @return double Covariance of X and Y
 * @throws std::invalid_argument if arrays are not 1D or lengths mismatch
 * @throws std::domain_error if arrays are empty or contain NaN/Inf
 */
template <typename T>
double cov(const ndarray<T>& X, const ndarray<T>& Y) {
    if (X.shape().size() != 1 || Y.shape().size() != 1)
        throw std::invalid_argument("cov: only 1D arrays supported");
    if (X.size() != Y.size())
        throw std::invalid_argument("cov: arrays must have same length");
    if (X.size() == 0)
        throw std::domain_error("cov: empty arrays");

    check_valid_numbers(X);
    check_valid_numbers(Y);

    double mean_x = mean(X);
    double mean_y = mean(Y);
    double c = 0.0;
    for (size_t i = 0; i < X.size(); ++i)
        c += (static_cast<double>(X.data()[i]) - mean_x) *
             (static_cast<double>(Y.data()[i]) - mean_y);
    return c / static_cast<double>(X.size());
}

/**
 * @brief Compute Pearson correlation coefficient between two 1D ndarrays.
 *
 * r = cov(X, Y) / (stddev(X) * stddev(Y))
 *
 * @tparam T Arithmetic type
 * @param X First 1D ndarray
 * @param Y Second 1D ndarray
 * @return double Correlation coefficient in [-1,1]
 * @throws std::domain_error if standard deviation of X or Y is zero
 * @throws std::invalid_argument if arrays are not 1D or lengths mismatch
 */
template <typename T>
double corrcoef(const ndarray<T>& X, const ndarray<T>& Y) {
    double denominator = stddev(X) * stddev(Y);
    if (denominator == 0.0)
        throw std::domain_error("corrcoef: division by zero (constant array)");
    return cov(X, Y) / denominator;
}

/**
 * @brief Compute histogram of a 1D ndarray.
 *
 * Divides range [min, max] into bins and counts elements in each bin.
 *
 * @tparam T Numeric type
 * @param A 1D ndarray
 * @param bins Number of bins (>=2)
 * @return std::vector<size_t> Bin counts
 * @throws std::invalid_argument if array is not 1D, empty, or bins < 2
 * @throws std::domain_error if all values are identical
 */
template <typename T>
std::vector<size_t> histogram(const ndarray<T>& A, size_t bins) {
    if (A.shape().size() != 1)
        throw std::invalid_argument("histogram: only 1D arrays supported");
    if (A.size() == 0)
        throw std::invalid_argument("histogram: empty array");
    if (bins < 2)
        throw std::invalid_argument("histogram: bins must be >= 2");

    check_valid_numbers(A);

    T min_val = *std::min_element(A.data().begin(), A.data().end());
    T max_val = *std::max_element(A.data().begin(), A.data().end());
    if (min_val == max_val)
        throw std::domain_error("histogram: all values identical, cannot create bins");

    std::vector<size_t> counts(bins, 0);
    double bin_width = static_cast<double>(max_val - min_val) / bins;

    for (const auto& v : A.data()) {
        size_t idx = static_cast<size_t>((v - min_val) / bin_width);
        if (idx == bins) idx = bins - 1; // include max
        counts[idx]++;
    }
    return counts;
}

/**
 * @brief Compute the percentile of a 1D ndarray.
 *
 * Returns the value below which p% of data fall.
 *
 * @tparam T Numeric type
 * @param A 1D ndarray
 * @param p Percentile (0 ≤ p ≤ 100)
 * @return double Percentile value
 * @throws std::invalid_argument if array is not 1D or p out of range
 * @throws std::domain_error if array is empty or contains NaN/Inf
 */
template <typename T>
double percentile(ndarray<T> A, double p) {
    if (A.shape().size() != 1)
        throw std::invalid_argument("percentile: only 1D arrays supported");
    if (A.size() == 0)
        throw std::domain_error("percentile: empty array");
    if (p < 0.0 || p > 100.0)
        throw std::invalid_argument("percentile: p must be in [0,100]");

    check_valid_numbers(A);

    std::vector<T> sorted = A.data();
    std::sort(sorted.begin(), sorted.end());

    double k = (p / 100.0) * (sorted.size() - 1);
    size_t f = static_cast<size_t>(std::floor(k));
    size_t c = static_cast<size_t>(std::ceil(k));
    if (f == c) return static_cast<double>(sorted[f]);
    return static_cast<double>(sorted[f]) * (c - k) +
           static_cast<double>(sorted[c]) * (k - f);
}

} // namespace numbits
