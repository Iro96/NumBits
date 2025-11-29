# NumBits

A comprehensive NumPy-like library written entirely in C++17. NumBits provides multidimensional arrays, mathematical operations, linear algebra, broadcasting, and more - all implemented purely in C++.

---

## Features

### 1. Core Functionality

- **Multidimensional Arrays**: N-dimensional array (tensor) support with configurable shapes
- **Memory Management**: Efficient memory management with move semantics and copy-on-write capabilities
- **Type Support**: Supports float, double, int32, int64, uint8, and bool types
- **Shape and Strides**: Efficient indexing using shape and stride information

### 2. Mathematical Operations

- **Element-wise Operations**: Addition, subtraction, multiplication, division
- **Scalar Operations**: Operations with scalar values
- **Comparison Operations**: Equal, not equal, less, greater, less_equal, greater_equal
- **Reduction Operations**: Sum, mean, min, max
- **Extrema Utilities**: Flat `argmax`/`argmin` helpers to retrieve indices
- **Value Clipping**: NumPy-style `clip` with support for scalar or broadcasted bounds
- **Logical Utilities**: `logical_and`, `logical_or`, `logical_xor`, `logical_not`, plus boolean reductions `all`/`any`
- **Cumulative Math**: `cumsum` and `cumprod` mirroring NumPy’s running operations

### 3. Broadcasting

- **Automatic Broadcasting**: NumPy-style broadcasting for operations between arrays of different shapes
- **Shape Compatibility**: Automatic shape expansion and compatibility checking

### 4. Mathematical Functions

- **Trigonometric**: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
- **Exponential/Logarithmic**: exp, log, log10
- **Power Functions**: pow, sqrt
- **Rounding**: ceil, floor, round
- **Other**: abs

### 5. Linear Algebra

- **Matrix Operations**: Matrix multiplication (matmul), dot product
- **Matrix Properties**: Transpose, determinant, inverse, trace
- **Vector Operations**: Vector dot product, matrix-vector multiplication

### 6. Array Manipulation

- **Reshaping**: Reshape arrays to different dimensions
- **Concatenation**: Concatenate arrays along specified axes
- **Stacking**: Stack arrays along new dimensions
- **Splitting**: Split arrays along axes
- **Tiling**: Repeat arrays to create larger arrays

### 7. Indexing and Slicing

- **Element Access**: Multi-dimensional indexing
- **Advanced Indexing**: Boolean indexing, advanced indexing
- **Slicing**: Extract subarrays using slicing operations

### 8. Array Creation

- **Sequence Generation**: `arange` for evenly spaced steps and `linspace` for fixed-length ranges with optional endpoints
- **Identity Matrices**: `eye` with optional rectangular shape and diagonal offset

---

## Building

### 1. Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15 or higher

### 2. Build Instructions

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Build examples (optional)
cmake --build . --target example_basic
cmake --build . --target example_linear_algebra
cmake --build . --target example_math
cmake --build . --target example_broadcasting

# Run examples
./examples/example_basic
./examples/example_linear_algebra
./examples/example_math
./examples/example_broadcasting
```

### 3. Installation

```bash
# From build directory
cmake --install . --prefix /path/to/install

# Or use default prefix
cmake --install .
```

---

## Usage

### How to compile

```bash
g++ /path/to/file.cpp -I../include -std=c++17 -o /path/to/output/file.exe
```
Then
```bash
/path/to/output/file.exe
```

### 1. Basic Example

```cpp
#include "numbits/array.hpp"
#include "numbits/operations.hpp"

using namespace numbits;

// Create arrays
Array<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
Array<float> b({2, 3}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

// Element-wise operations
auto c = a + b;
auto d = a * 2.0f;

// Create special arrays
auto zeros = Array<float>::zeros({3, 4});
auto ones = Array<float>::ones({2, 2});

// Reshape
auto flattened = a.flatten();
auto reshaped = flattened.reshape({3, 2});
```

### 2. Linear Algebra Example

```cpp
#include "numbits/array.hpp"
#include "numbits/linear_algebra.hpp"

using namespace numbits;

// Matrix multiplication
Array<float> A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
Array<float> B({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
auto C = matmul(A, B);

// Transpose
auto At = transpose(A);

// Determinant and inverse
Array<float> matrix({2, 2}, {4.0f, 7.0f, 2.0f, 6.0f});
float det = determinant(matrix);
auto inv = inverse(matrix);
```

### 3. Mathematical Functions Example

```cpp
#include "numbits/array.hpp"
#include "numbits/math_functions.hpp"

using namespace numbits;

Array<float> arr({2, 2}, {0.0f, M_PI/4.0f, M_PI/2.0f, M_PI});

// Trigonometric functions
auto sin_arr = sin(arr);
auto cos_arr = cos(arr);

// Exponential and logarithmic
auto exp_arr = exp(arr);
auto log_arr = log(exp_arr);

// Power and square root
auto sqrt_arr = sqrt(arr);
auto pow_arr = pow(arr, 2.0f);
```

### 4. Broadcasting Example

```cpp
#include "numbits/array.hpp"
#include "numbits/operations.hpp"

using namespace numbits;

// Broadcasting
Array<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
Array<float> b({3}, {10.0f, 20.0f, 30.0f});

// Automatic broadcasting
auto c = a + b;  // b is broadcasted to match a's shape

// Clip values similar to NumPy's np.clip
auto clamped = clip(c, 0.0f, 25.0f);

// Conditional blend
ndarray<bool> mask({2, 3}, {true, false, true, false, true, false});
auto blended = where(mask, c, clamped);

// Locate the flat index of the largest entry
auto max_idx = argmax(blended);

// Check conditions and running totals
auto positives = logical_and(blended, blended);
bool any_large = any(blended > 20.0f);
auto partial_sums = cumsum(blended);
```

---

## API Reference

### 1. Array Class

```cpp
template<typename T>
class Array {
    // Constructors
    Array(const Shape& shape);
    Array(const Shape& shape, const std::vector<T>& data);
    
    // Accessors
    const Shape& shape() const;
    const Strides& strides() const;
    size_t size() const;
    size_t ndim() const;
    T* data();
    const T* data() const;
    
    // Element access
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    
    // Static factory methods
    static Array zeros(const Shape& shape);
    static Array ones(const Shape& shape);
    static Array full(const Shape& shape, const T& value);
    
    // Manipulation
    Array reshape(const Shape& new_shape) const;
    Array flatten() const;
};
```

### 2. Operations

```cpp
// Element-wise operations
template<typename T> Array<T> add(const Array<T>& a, const Array<T>& b);
template<typename T> Array<T> subtract(const Array<T>& a, const Array<T>& b);
template<typename T> Array<T> multiply(const Array<T>& a, const Array<T>& b);
template<typename T> Array<T> divide(const Array<T>& a, const Array<T>& b);

// Scalar operations
template<typename T> Array<T> add_scalar(const Array<T>& a, T scalar);
template<typename T> Array<T> multiply_scalar(const Array<T>& a, T scalar);

// Reduction operations
template<typename T> T sum(const Array<T>& arr);
template<typename T> T mean(const Array<T>& arr);
template<typename T> T min(const Array<T>& arr);
template<typename T> T max(const Array<T>& arr);

// Operator overloads
template<typename T> Array<T> operator+(const Array<T>& a, const Array<T>& b);
template<typename T> Array<T> operator*(const Array<T>& a, T scalar);
// ... and more
```

### 3. Linear Algebra

```cpp
template<typename T> Array<T> matmul(const Array<T>& a, const Array<T>& b);
template<typename T> Array<T> dot(const Array<T>& a, const Array<T>& b);
template<typename T> Array<T> transpose(const Array<T>& arr);
template<typename T> T determinant(const Array<T>& arr);
template<typename T> Array<T> inverse(const Array<T>& arr);
template<typename T> T trace(const Array<T>& arr);
```

### 4. Math Functions

```cpp
template<typename T> Array<T> sin(const Array<T>& arr);
template<typename T> Array<T> cos(const Array<T>& arr);
template<typename T> Array<T> exp(const Array<T>& arr);
template<typename T> Array<T> log(const Array<T>& arr);
template<typename T> Array<T> sqrt(const Array<T>& arr);
template<typename T> Array<T> pow(const Array<T>& arr, T exponent);
// ... and more
```

---

## Performance

NumBits is designed for performance:

- Efficient memory layout using strides
- Move semantics to avoid unnecessary copies
- Template-based implementation for zero-cost abstractions
- Direct memory access for optimal performance

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Future Enhancements

- GPU acceleration support (CUDA/OpenCL)
- Parallel operations using OpenMP or threading
- More linear algebra operations (SVD, eigendecomposition, etc.)
- More array manipulation functions
- Better broadcasting performance
- SIMD optimizations
- Sparse array support

