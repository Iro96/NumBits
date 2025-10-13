<div align="center">
  
# NumBits

NumBits is a lightweight C++ numerical computing library inspired by NumPy. It provides multidimensional arrays (`ndarray<T>`), basic arithmetic operations, reductions, linear algebra, random number generation, and statistical functions.

[![CMake on multiple platforms](https://github.com/Iro96/NumBits/actions/workflows/cmake-multi-platform.yml/badge.svg?branch=main)](https://github.com/Iro96/NumBits/actions/workflows/cmake-multi-platform.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

---

## NumBits Development Plan (v0.1 → v1.0+)

| Version      | Stage                | Focus                          | Description                                                          |
| ------------ | -------------------- | ------------------------------ | -------------------------------------------------------------------- |
| ✅ **v0.1**  | Core MVP             | Array + Basic Ops              | `ndarray<T>`, arithmetic, reductions, random, dot, stats             |
| ✅ **v0.2**  | Array Manipulation   | Shape ops, broadcasting, views | `reshape`, `transpose`, `expand_dims`, `broadcast_to`, slicing       |
| 🔜 **v0.3**  | Linear Algebra       | Full matrix API                | `matmul`, `inv`, `det`, `eig`, `svd`, `norm`, `trace`                |
| 🔜 **v0.4**  | Advanced Math        | Universal functions            | `sin`, `cos`, `tan`, `log`, `exp`, `pow`, elementwise `ufunc` system |
| 🔜 **v0.5**  | Statistics           | Correlation, covariance        | `corrcoef`, `cov`, `histogram`, `percentile`                         |
| 🔜 **v0.6**  | I/O + Serialization  | File save/load                 | `save`, `load`, `savetxt`, `loadtxt`, binary `.nbc` support          |
| 🔜 **v0.7**  | Random 2.0           | Full RNG distributions         | `normal`, `uniform`, `poisson`, `choice`, seeding                    |
| 🔜 **v0.8**  | Backend Acceleration | BLAS / SIMD                    | optional Eigen / OpenBLAS backend, parallel reductions               |
| 🔜 **v0.9**  | GPU Backend          | CUDA / OpenCL / Vulkan         | GPU-enabled ndarray backend                                          |
| 🔜 **v1.0**  | Python API           | `import numbits` via pybind11  | Expose C++ API to Python for hybrid workflows                        |
| 🌟 **v1.1+** | AI / Autograd        | Differentiation engine         | Automatic gradients, neural ops, backpropagation                     |

---

## How To Use

### How to compile

#### Clone the repository

```bah
git clone https://github.com/Iro96/NumBits.git
cd ./NumBits
```

#### Windows

```bash
g++ path_to_your_file.cpp -o path_to_your_output -I"d:\NumBits\include" -std=c++20
path_to_your_output
```

### Example code (v0.2)

```cpp
#include <iostream>
#include "numbits/core/ndarray.hpp"
#include "numbits/core/reshape.hpp"
#include "numbits/ops/arithmetic.hpp"
#include "numbits/ops/reduction.hpp"
#include "numbits/linalg/matrix.hpp"
#include "numbits/math/math.hpp"
#include "numbits/stats/statistics.hpp"
#include "numbits/random/generator.hpp"

int main() {
    using namespace numbits;

    std::cout << "=== NumBits v0.2 Full Example ===\n\n";

    // 1. Create arrays
    ndarray<double> A({2, 3}, 2.0);          // 2x3 filled with 2.0
    ndarray<double> B = rand<double>({2, 3}); // uniform random
    ndarray<double> C = randn<double>({2, 3}); // normal random

    std::cout << "Array A:\n" << A << "\n";
    std::cout << "Array B (uniform):\n" << B << "\n";
    std::cout << "Array C (normal):\n" << C << "\n";

    // 2. Elementwise arithmetic
    auto D = add(A, B);
    auto E = sub(B, C);
    auto F = mul(A, B);
    auto G = div(B, A);

    std::cout << "D = A + B:\n" << D << "\n";
    std::cout << "E = B - C:\n" << E << "\n";
    std::cout << "F = A * B:\n" << F << "\n";
    std::cout << "G = B / A:\n" << G << "\n";

    // 3. Linear algebra (dot product)
    ndarray<double> X({2, 3}, 1.0);
    ndarray<double> Y({3, 2}, 2.0);
    auto Z = dot(X, Y);

    std::cout << "Z = X dot Y:\n" << Z << "\n";

    // 4. Math functions (elementwise)
    auto H = exp(B);
    auto I = sqrt(C * C + 1.0);  // ensure positive input for sqrt

    std::cout << "H = exp(B):\n" << H << "\n";
    std::cout << "I = sqrt(C^2 + 1):\n" << I << "\n";

    // 5. Reductions
    std::cout << "sum(B) = " << sum(B) << "\n";
    std::cout << "mean(B) = " << mean(B) << "\n";
    std::cout << "variance(B) = " << variance(B) << "\n";
    std::cout << "stddev(B) = " << stddev(B) << "\n";

    // 6. Reshape & broadcasting
    auto R = reshape(A, {3, 2});  // 2x3 -> 3x2
    std::cout << "Reshaped A (3x2):\n" << R << "\n";

    auto S = expand_dims(B, 0); // add new axis at front -> shape (1,2,3)
    std::cout << "Expanded B shape (1x2x3): " << S.shape()[0] << "x" << S.shape()[1] << "x" << S.shape()[2] << "\n";

    auto T = broadcast_to(A, {4, 2, 3}); // broadcast 2x3 -> 4x2x3
    std::cout << "Broadcasted A shape (2x3x4): " << T.shape()[0] << "x" << T.shape()[1] << "x" << T.shape()[2] << "\n";

    // 7. Squeeze
    auto U = squeeze(S); // remove size-1 axes -> back to 2x3
    std::cout << "Squeezed array shape: " << U.shape()[0] << "x" << U.shape()[1] << "\n";

    // 8. Slice (2D)
    auto V = slice(A, 0, 2, 1, 3); // take columns 1..2
    std::cout << "Slice of A (columns 1..2):\n" << V << "\n";

    // 9. Random arrays
    auto U_rand = rand<double>({3, 3});
    auto N_rand = randn<double>({3, 3});

    std::cout << "U_rand (uniform 3x3):\n" << U_rand << "\n";
    std::cout << "N_rand (normal 3x3):\n" << N_rand << "\n";

    return 0;
}
```
