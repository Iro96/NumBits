#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <array>
#include <iterator>

namespace numbits {

/**
 * @brief Computes the sum of all elements in a numeric vector.
 *
 * This function iterates through all elements in the input vector and returns
 * their accumulated sum. The accumulation type is promoted automatically:
 * - For integral types (`int`, `long`, etc.), accumulation is done using `long long`.
 * - For floating-point types (`float`, `double`), accumulation is done using `double`.
 *
 * The function requires that the element type `T` satisfies `std::is_arithmetic_v<T>`.
 *
 * Example:
 * @code
 * std::vector<int> v = {1, 2, 3, 4};
 * auto total = numbits::sum(v); // total == 10 (type: long long)
 *
 * std::vector<double> d = {1.5, 2.5};
 * auto total_d = numbits::sum(d); // total_d == 4.0 (type: double)
 * @endcode
 *
 * @tparam T Element type of the vector (must be arithmetic).
 * @param data Input vector of numeric values.
 * @return The total sum of elements, promoted to `long long` or `double` depending on `T`.
 *
 * @throws std::logic_error Never throws (no runtime exceptions).
 */
template <typename T>
auto sum(const std::vector<T>& data) {
    static_assert(std::is_arithmetic_v<T>, "sum() requires arithmetic type");
    using AccT = std::conditional_t<std::is_integral_v<T>, long long, double>;
    return std::accumulate(data.begin(), data.end(), AccT(0));
}

/**
 * @brief Computes the arithmetic mean (average) of all elements in a numeric vector.
 *
 * The mean is defined as the sum of all elements divided by the number of elements.
 * The result is returned as a `double`, ensuring floating-point precision even
 * when the input type `T` is integral.
 *
 * If the input vector is empty, the function returns `0.0` to avoid division by zero.
 *
 * Example:
 * @code
 * std::vector<int> v = {1, 2, 3, 4};
 * double avg = numbits::mean(v); // avg == 2.5
 *
 * std::vector<double> d = {1.2, 3.8, 5.0};
 * double avg_d = numbits::mean(d); // avg_d == 3.333333...
 *
 * std::vector<int> empty;
 * double avg_empty = numbits::mean(empty); // avg_empty == 0.0
 * @endcode
 *
 * @tparam T Element type of the vector (must be arithmetic).
 * @param data Input vector of numeric values.
 * @return The mean (average) of the input values as a double.
 *
 * @note This function performs floating-point division even for integer vectors.
 * @note For empty vectors, returns `0.0` instead of throwing an exception.
 *
 * @throws std::logic_error Never throws (no runtime exceptions).
 */
template <typename T>
double mean(const std::vector<T>& data) {
    if (data.empty()) return 0.0;
    return static_cast<double>(sum(data)) / static_cast<double>(data.size());
}

/**
 * @brief A small vector optimization class that uses std::array for small sizes and std::vector for larger sizes.
 *
 * This class provides a vector-like interface but optimizes memory usage for small sizes (up to N elements)
 * by using std::array instead of dynamic allocation. For sizes > N, it falls back to std::vector.
 *
 * @tparam T The element type.
 * @tparam N The maximum size for small vector optimization (default 8).
 */
template <typename T, size_t N = 8>
class small_vector {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    // Default constructor
    small_vector() : size_(0), is_small_(true) {}

    // Constructor with size and value
    explicit small_vector(size_type count, const T& value = T()) : size_(count) {
        if (count <= N) {
            is_small_ = true;
            std::fill_n(small_data_.begin(), count, value);
        } else {
            is_small_ = false;
            large_data_ = std::vector<T>(count, value);
        }
    }

    // Constructor from initializer list
    small_vector(std::initializer_list<T> init) : size_(init.size()) {
        if (init.size() <= N) {
            is_small_ = true;
            std::copy(init.begin(), init.end(), small_data_.begin());
        } else {
            is_small_ = false;
            large_data_ = std::vector<T>(init);
        }
    }

    // Constructor from range
    template <typename InputIt>
    small_vector(InputIt first, InputIt last) {
        size_type count = std::distance(first, last);
        size_ = count;
        if (count <= N) {
            is_small_ = true;
            std::copy(first, last, small_data_.begin());
        } else {
            is_small_ = false;
            large_data_ = std::vector<T>(first, last);
        }
    }

    // Copy constructor
    small_vector(const small_vector& other) : size_(other.size_), is_small_(other.is_small_) {
        if (is_small_) {
            new (&small_data_) std::array<T, N>(other.small_data_);
        } else {
            new (&large_data_) std::vector<T>(other.large_data_);
        }
    }

    // Move constructor
    small_vector(small_vector&& other) noexcept : size_(other.size_), is_small_(other.is_small_) {
        if (is_small_) {
            new (&small_data_) std::array<T, N>(std::move(other.small_data_));
        } else {
            new (&large_data_) std::vector<T>(std::move(other.large_data_));
        }
        other.size_ = 0;
        other.is_small_ = true;
    }

    // Assignment operators
    small_vector& operator=(const small_vector& other) {
        if (this != &other) {
            // Clean up current state
            this->~small_vector();
            size_ = other.size_;
            is_small_ = other.is_small_;
            if (is_small_) {
                new (&small_data_) std::array<T, N>(other.small_data_);
            } else {
                new (&large_data_) std::vector<T>(other.large_data_);
            }
        }
        return *this;
    }

    small_vector& operator=(small_vector&& other) noexcept {
        if (this != &other) {
            // Clean up current state
            this->~small_vector();
            size_ = other.size_;
            is_small_ = other.is_small_;
            if (is_small_) {
                new (&small_data_) std::array<T, N>(std::move(other.small_data_));
            } else {
                new (&large_data_) std::vector<T>(std::move(other.large_data_));
            }
            other.size_ = 0;
            other.is_small_ = true;
        }
        return *this;
    }

    // Destructor
    ~small_vector() {
        if (is_small_) {
            small_data_.~array();
        } else {
            large_data_.~vector();
        }
    }

    // Element access
    reference operator[](size_type pos) {
        return is_small_ ? small_data_[pos] : large_data_[pos];
    }

    const_reference operator[](size_type pos) const {
        return is_small_ ? small_data_[pos] : large_data_[pos];
    }

    reference at(size_type pos) {
        if (pos >= size_) throw std::out_of_range("small_vector::at");
        return (*this)[pos];
    }

    const_reference at(size_type pos) const {
        if (pos >= size_) throw std::out_of_range("small_vector::at");
        return (*this)[pos];
    }

    reference front() { return (*this)[0]; }
    const_reference front() const { return (*this)[0]; }
    reference back() { return (*this)[size_ - 1]; }
    const_reference back() const { return (*this)[size_ - 1]; }

    // Iterators
    iterator begin() { return is_small_ ? small_data_.data() : large_data_.data(); }
    const_iterator begin() const { return is_small_ ? small_data_.data() : large_data_.data(); }
    const_iterator cbegin() const { return begin(); }

    iterator end() { return is_small_ ? small_data_.data() + size_ : large_data_.data() + size_; }
    const_iterator end() const { return is_small_ ? small_data_.data() + size_ : large_data_.data() + size_; }
    const_iterator cend() const { return end(); }

    // Capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type max_size() const noexcept { return is_small_ ? N : large_data_.max_size(); }

    // Modifiers
    void clear() noexcept {
        size_ = 0;
        is_small_ = true;
    }

    void push_back(const T& value) {
        if (is_small_ && size_ < N) {
            small_data_[size_] = value;
        } else if (is_small_ && size_ == N) {
            // Convert to large
            std::vector<T> temp(small_data_.begin(), small_data_.end());
            temp.push_back(value);
            large_data_ = std::move(temp);
            is_small_ = false;
        } else {
            large_data_.push_back(value);
        }
        ++size_;
    }

    void push_back(T&& value) {
        if (is_small_ && size_ < N) {
            small_data_[size_] = std::move(value);
        } else if (is_small_ && size_ == N) {
            // Convert to large
            std::vector<T> temp(small_data_.begin(), small_data_.end());
            temp.push_back(std::move(value));
            large_data_ = std::move(temp);
            is_small_ = false;
        } else {
            large_data_.push_back(std::move(value));
        }
        ++size_;
    }

    void pop_back() {
        if (size_ > 0) {
            --size_;
            if (!is_small_ && size_ <= N) {
                // Convert back to small
                std::array<T, N> temp;
                std::copy(large_data_.begin(), large_data_.end(), temp.begin());
                small_data_ = temp;
                is_small_ = true;
            }
        }
    }

    void resize(size_type count) {
        resize(count, T());
    }

    void resize(size_type count, const T& value) {
        if (count <= N) {
            if (!is_small_) {
                // Convert to small
                std::array<T, N> temp;
                size_type copy_count = std::min(size_, count);
                std::copy(large_data_.begin(), large_data_.begin() + copy_count, temp.begin());
                if (copy_count < count) {
                    std::fill(temp.begin() + copy_count, temp.begin() + count, value);
                }
                small_data_ = temp;
                is_small_ = true;
            } else {
                if (count > size_) {
                    std::fill(small_data_.begin() + size_, small_data_.begin() + count, value);
                }
            }
        } else {
            if (is_small_) {
                // Convert to large
                large_data_ = std::vector<T>(small_data_.begin(), small_data_.begin() + size_);
                large_data_.resize(count, value);
                is_small_ = false;
            } else {
                large_data_.resize(count, value);
            }
        }
        size_ = count;
    }

    // Comparison operators
    bool operator==(const small_vector& other) const {
        if (size_ != other.size_) return false;
        for (size_type i = 0; i < size_; ++i) {
            if ((*this)[i] != other[i]) return false;
        }
        return true;
    }

    bool operator!=(const small_vector& other) const {
        return !(*this == other);
    }

private:
    size_type size_;
    bool is_small_;
    union {
        std::array<T, N> small_data_;
        std::vector<T> large_data_;
    };
};

} // namespace numbits
