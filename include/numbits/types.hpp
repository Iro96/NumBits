/**
 * @file types.hpp
 * @brief Core type definitions and utilities for NumBits arrays.
 *
 * This header defines fundamental types for n-dimensional arrays (ndarrays) in NumBits:
 *   - Index and size types
 *   - Shape and strides representations
 *   - DType enum for supported data types
 *   - Compile-time utilities for mapping between C++ types and DType
 *
 * @namespace numbits
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>

namespace numbits {

/**
 * @brief Signed index type used for array indexing.
 */
using index_t = std::ptrdiff_t;

/**
 * @brief Unsigned size type used for array sizes and dimensions.
 */
using size_t = std::size_t;

/**
 * @brief Enumeration of supported data types for ndarrays.
 */
enum class DType {
    FLOAT32,  ///< 32-bit floating point
    FLOAT64,  ///< 64-bit floating point
    INT32,    ///< 32-bit signed integer
    INT64,    ///< 64-bit signed integer
    UINT8,    ///< 8-bit unsigned integer
    UINT16,   ///< 16-bit unsigned integer
    UINT32,   ///< 32-bit unsigned integer
    UINT64,   ///< 64-bit unsigned integer
    BOOL      ///< Boolean type
};

/**
 * @brief Compile-time mapping from C++ type to DType.
 *
 * Example:
 * @code
 * DType dt = dtype_from_type<float>(); // returns DType::FLOAT32
 * @endcode
 *
 * @tparam T C++ type
 * @return Corresponding DType enumeration
 * @throws static_assert for unsupported types
 */
template<typename T>
constexpr DType dtype_from_type() {
    if constexpr (std::is_same_v<T, float>) return DType::FLOAT32;
    else if constexpr (std::is_same_v<T, double>) return DType::FLOAT64;
    else if constexpr (std::is_same_v<T, int32_t>) return DType::INT32;
    else if constexpr (std::is_same_v<T, int64_t>) return DType::INT64;
    else if constexpr (std::is_same_v<T, uint8_t>) return DType::UINT8;
    else if constexpr (std::is_same_v<T, uint16_t>) return DType::UINT16;
    else if constexpr (std::is_same_v<T, uint32_t>) return DType::UINT32;
    else if constexpr (std::is_same_v<T, uint64_t>) return DType::UINT64;
    else if constexpr (std::is_same_v<T, bool>) return DType::BOOL;
    else static_assert(std::is_same_v<T, void>, "Unsupported type for dtype_from_type");
}

/**
 * @brief Compile-time mapping from DType to C++ type.
 *
 * Usage:
 * @code
 * using T = dtype_to_type<DType::FLOAT32>::type; // T is float
 * @endcode
 *
 * @tparam dtype DType enumeration
 */
template<DType dtype>
struct dtype_to_type;  // Primary template left undefined

// Specializations for each supported type
template<> struct dtype_to_type<DType::FLOAT32> { using type = float; };
template<> struct dtype_to_type<DType::FLOAT64> { using type = double; };
template<> struct dtype_to_type<DType::INT32>   { using type = int32_t; };
template<> struct dtype_to_type<DType::INT64>   { using type = int64_t; };
template<> struct dtype_to_type<DType::UINT8>   { using type = uint8_t; };
template<> struct dtype_to_type<DType::UINT16>  { using type = uint16_t; };
template<> struct dtype_to_type<DType::UINT32>  { using type = uint32_t; };
template<> struct dtype_to_type<DType::UINT64>  { using type = uint64_t; };
template<> struct dtype_to_type<DType::BOOL>    { using type = bool; };

/**
 * @brief Represents the shape of an ndarray (number of elements in each dimension).
 *
 * Example: A 3x4 array has shape {3, 4}.
 */
using Shape = std::vector<size_t>;

/**
 * @brief Represents the strides of an ndarray (step in memory for each dimension).
 *
 * Strides are used for indexing and broadcasting calculations.
 */
using Strides = std::vector<size_t>;

} // namespace numbits
