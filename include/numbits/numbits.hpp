/**
 * @file numbits.hpp
 * @brief Main header file for NumBits library.
 *
 * This is the primary include file that brings in all NumBits functionality:
 *   - Core ndarray class and types
 *   - Element-wise and reduction operations
 *   - Broadcasting utilities
 *   - Mathematical functions
 *   - Linear algebra operations
 *   - Array manipulation (concatenate, stack, split, tile)
 *   - Array creation utilities (arange, linspace, eye)
 *   - Advanced indexing and slicing
 *   - Random number generation
 *   - File I/O (text and binary)
 *
 * @example
 * @code
 *   #include "numbits/numbits.hpp"
 *   using namespace numbits;
 *
 *   int main() {
 *       // Create arrays
 *       ndarray<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
 *       ndarray<float> b({2, 3}, {7, 8, 9, 10, 11, 12});
 *
 *       // Perform operations
 *       auto c = a + b;
 *       auto d = a * 2.0f;
 *
 *       // Linear algebra
 *       auto At = transpose(a);
 *
 *       return 0;
 *   }
 * @endcode
 *
 * @namespace numbits
 */

#pragma once

// Main header file for NumBits - includes all functionality

#include "numbits/ndarray.hpp"
#include "numbits/types.hpp"
#include "numbits/utils.hpp"
#include "numbits/operations.hpp"
#include "numbits/broadcasting.hpp"
#include "numbits/math_functions.hpp"
#include "numbits/linear_algebra.hpp"
#include "numbits/ndarray_manipulation.hpp"
#include "numbits/creation.hpp"
#include "numbits/indexing.hpp"
#include "numbits/random.hpp"
#include "numbits/io.hpp"

// Convenience namespace
namespace nb = numbits;
