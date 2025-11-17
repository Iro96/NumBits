#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace numbits {

using index_t = std::ptrdiff_t;
using size_t = std::size_t;

enum class DType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8,
    BOOL
};

template<typename T>
constexpr DType dtype_from_type() {
    if constexpr (std::is_same_v<T, float>) return DType::FLOAT32;
    else if constexpr (std::is_same_v<T, double>) return DType::FLOAT64;
    else if constexpr (std::is_same_v<T, int32_t>) return DType::INT32;
    else if constexpr (std::is_same_v<T, int64_t>) return DType::INT64;
    else if constexpr (std::is_same_v<T, uint8_t>) return DType::UINT8;
    else if constexpr (std::is_same_v<T, bool>) return DType::BOOL;
    else static_assert(std::is_same_v<T, void>, "Unsupported type");
}

template<DType dtype>
struct dtype_to_type;

template<> struct dtype_to_type<DType::FLOAT32> { using type = float; };
template<> struct dtype_to_type<DType::FLOAT64> { using type = double; };
template<> struct dtype_to_type<DType::INT32> { using type = int32_t; };
template<> struct dtype_to_type<DType::INT64> { using type = int64_t; };
template<> struct dtype_to_type<DType::UINT8> { using type = uint8_t; };
template<> struct dtype_to_type<DType::BOOL> { using type = bool; };

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

} // namespace numbits

