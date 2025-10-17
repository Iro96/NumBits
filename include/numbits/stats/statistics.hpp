#pragma once
#include "../core/ndarray.hpp"
#include "../ops/reduction.hpp"  // for mean()
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
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
 * @brief Convert a 1D or 2D ndarray to std::vector<std::vector<double>>.
 *
 * 1D arrays become a single row.
 * 2D arrays become row-major vectors.
 *
 * @tparam T Numeric type
 * @param A Input ndarray
 * @return 2D vector representation
 * @throws std::invalid_argument if A is not 1D or 2D
 */
template <typename T>
std::vector<std::vector<double>> to_matrix(const ndarray<T>& A) {
    std::vector<std::vector<double>> mat;
    if (A.shape().size() == 1) {
        mat.push_back(std::vector<double>(A.data().begin(), A.data().end()));
    } else if (A.shape().size() == 2) {
        size_t n_cols = A.shape()[1];
        for (size_t r = 0; r < A.shape()[0]; ++r) {
            std::vector<double> row(n_cols);
            for (size_t c = 0; c < n_cols; ++c)
                row[c] = static_cast<double>(A(r, c));
            mat.push_back(row);
        }
    } else {
        throw std::invalid_argument("to_matrix: input must be 1D or 2D");
    }
    return mat;
}

/**
 * @brief Compute the population variance of an n-D ndarray.
 *
 * Computes σ² = (1/N) ∑(x_i - mean)² over all elements.
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
 * @brief Compute the population standard deviation of an n-D ndarray.
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
 * @brief Compute covariance matrix of two ndarrays (1D or 2D).
 *
 * If X is 1D, treated as a single variable.
 * If X is 2D, each row is a variable, each column an observation.
 *
 * @tparam T Numeric type
 * @param X First array (1D or 2D)
 * @param Y Second array (same shape as X)
 * @return ndarray<double> Covariance matrix
 * @throws std::invalid_argument for shape mismatch or empty input
 * @throws std::domain_error if any NaN/Inf
 */
template <typename T>
ndarray<double> cov(const ndarray<T>& X, const ndarray<T>& Y) {
    auto Xmat = to_matrix(X);
    auto Ymat = to_matrix(Y);

    size_t n_obs = Xmat[0].size();
    if (Ymat.size() != Xmat.size() || Ymat[0].size() != n_obs)
        throw std::invalid_argument("cov: X and Y shapes must match");

    // Validate numbers
    for (auto& row : Xmat) for (double v : row) if (!std::isfinite(v)) throw std::domain_error("cov: NaN/Inf in X");
    for (auto& row : Ymat) for (double v : row) if (!std::isfinite(v)) throw std::domain_error("cov: NaN/Inf in Y");

    size_t n_vars_X = Xmat.size();
    size_t n_vars_Y = Ymat.size();
    ndarray<double> C({n_vars_X, n_vars_Y}, 0.0);

    for (size_t i = 0; i < n_vars_X; ++i) {
        double mean_X = std::accumulate(Xmat[i].begin(), Xmat[i].end(), 0.0) / n_obs;
        for (size_t j = 0; j < n_vars_Y; ++j) {
            double mean_Y = std::accumulate(Ymat[j].begin(), Ymat[j].end(), 0.0) / n_obs;
            double cov_ij = 0.0;
            for (size_t k = 0; k < n_obs; ++k)
                cov_ij += (Xmat[i][k] - mean_X) * (Ymat[j][k] - mean_Y);
            C(i, j) = cov_ij / static_cast<double>(n_obs);
        }
    }
    return C;
}

/**
 * @brief Compute covariance matrix of a single ndarray (X with itself).
 */
template <typename T>
ndarray<double> cov(const ndarray<T>& X) {
    return cov(X, X);
}

/**
 * @brief Compute Pearson correlation coefficient matrix for 1D or 2D ndarrays.
 *
 * r_ij = cov(X_i, X_j) / (stddev(X_i) * stddev(X_j))
 *
 * @tparam T Numeric type
 * @param X First array (1D or 2D)
 * @param Y Second array (same shape as X)
 * @return ndarray<double> Correlation coefficient matrix
 * @throws std::domain_error if any stddev = 0
 */
template <typename T>
ndarray<double> corrcoef(const ndarray<T>& X, const ndarray<T>& Y) {
    ndarray<double> C = cov(X, Y);
    auto Xmat = to_matrix(X);
    size_t n_vars = Xmat.size();
    std::vector<double> stds(n_vars);

    for (size_t i = 0; i < n_vars; ++i) {
        double mean_i = std::accumulate(Xmat[i].begin(), Xmat[i].end(), 0.0) / Xmat[i].size();
        double var_i = 0.0;
        for (double v : Xmat[i]) var_i += (v - mean_i) * (v - mean_i);
        stds[i] = std::sqrt(var_i / static_cast<double>(Xmat[i].size()));
        if (stds[i] == 0.0)
            throw std::domain_error("corrcoef: division by zero (constant row)");
    }

    for (size_t i = 0; i < n_vars; ++i)
        for (size_t j = 0; j < n_vars; ++j)
            C(i, j) /= (stds[i] * stds[j]);

    return C;
}

/**
 * @brief Compute correlation coefficient matrix of a single ndarray (X with itself).
 */
template <typename T>
ndarray<double> corrcoef(const ndarray<T>& X) {
    return corrcoef(X, X);
}

/**
 * @brief Compute histogram of an n-D ndarray.
 *
 * Flattens the array, divides the data range into bins, and counts the number
 * of elements in each bin.
 *
 * @tparam T Numeric type
 * @param A n-D ndarray
 * @param bins Number of bins (>=2)
 * @return std::pair<ndarray<size_t>, ndarray<double>> Counts and edges of bins
 * @throws std::invalid_argument if array is empty or bins < 2
 * @throws std::domain_error if all values are identical
 */
template <typename T>
std::pair<ndarray<size_t>, ndarray<double>> histogram(const ndarray<T>& A, size_t bins) {
    if (A.size() == 0)
        throw std::invalid_argument("histogram: empty array");
    if (bins < 2)
        throw std::invalid_argument("histogram: bins must be >= 2");

    check_valid_numbers(A);

    double min_val = static_cast<double>(*std::min_element(A.data().begin(), A.data().end()));
    double max_val = static_cast<double>(*std::max_element(A.data().begin(), A.data().end()));
    if (min_val == max_val)
        throw std::domain_error("histogram: all values identical, cannot create bins");

    ndarray<size_t> counts({bins}, 0);
    ndarray<double> edges({bins + 1}, 0.0);

    double bin_width = (max_val - min_val) / bins;
    for (size_t i = 0; i <= bins; ++i)
        edges[i] = min_val + i * bin_width;

    for (const auto& v : A.data()) {
        size_t idx = static_cast<size_t>((v - min_val) / bin_width);
        if (idx == bins) idx = bins - 1;
        counts[idx]++;
    }

    return {counts, edges};
}

/**
 * @brief Compute the percentile of an n-D ndarray.
 *
 * Flattens the array and returns the value below which p% of data fall.
 *
 * @tparam T Numeric type
 * @param A n-D ndarray
 * @param p Percentile (0 ≤ p ≤ 100)
 * @return double Percentile value
 * @throws std::invalid_argument if p out of range
 * @throws std::domain_error if array is empty or contains NaN/Inf
 */
template <typename T>
double percentile(ndarray<T> A, double p) {
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
