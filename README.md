<div align="center">
  
# NumBits

NumBits is a lightweight C++ numerical computing library inspired by NumPy. It provides multidimensional arrays (`ndarray<T>`), basic arithmetic operations, reductions, linear algebra, random number generation, and statistical functions.

[![CMake on multiple platforms](https://github.com/Iro96/NumBits/actions/workflows/cmake-multi-platform.yml/badge.svg?branch=main)](https://github.com/Iro96/NumBits/actions/workflows/cmake-multi-platform.yml)
[![CodeQL Advanced](https://github.com/Iro96/NumBits/actions/workflows/codeql.yml/badge.svg)](https://github.com/Iro96/NumBits/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

---

## NumBits Development Plan (v0.1 → v1.0+)

| Version      | Stage                | Focus                          | Description                                                          |
| ------------ | -------------------- | ------------------------------ | -------------------------------------------------------------------- |
| ✅ **v0.1**  | Core MVP             | Array + Basic Ops              | `ndarray<T>`, arithmetic, reductions, random, dot, stats             |
| ✅ **v0.2**  | Array Manipulation   | Shape ops, broadcasting, views | `reshape`, `transpose`, `expand_dims`, `broadcast_to`, slicing       |
| ✅ **v0.3**  | Linear Algebra       | Full matrix API                | `matmul`, `inv`, `det`, `eig`, `svd`, `norm`, `trace`                |
| ✅ **v0.4**  | Advanced Math        | Universal functions            | `sin`, `cos`, `tan`, `log`, `exp`, `pow`, elementwise `ufunc` system |
| 🔜 **v0.5**  | Statistics           | Correlation, covariance        | `corrcoef`, `cov`, `histogram`, `percentile`                         |
| 🔜 **v0.6**  | I/O + Serialization  | File save/load                 | `save`, `load`, `savetxt`, `loadtxt`, binary `.nbc` support          |
| 🔜 **v0.7**  | Random 2.0           | Full RNG distributions         | `normal`, `uniform`, `poisson`, `choice`, seeding                    |
| 🔜 **v0.8**  | Backend Acceleration | BLAS / SIMD                    | optional Eigen / OpenBLAS backend, parallel reductions               |
| 🔜 **v0.9**  | GPU Backend          | CUDA / OpenCL / Vulkan         | GPU-enabled ndarray backend                                          |
| 🔜 **v1.0**  | Python API           | `import numbits` via pybind11  | Expose C++ API to Python for hybrid workflows                        |
| 🌟 **v1.1+** | AI / Autograd        | Differentiation engine         | Automatic gradients, neural ops, backpropagation                     |

### Contributing

We welcome contributions from everyone! For detailed guidelines on how to contribute, please see [CONTRIBUTING.md](https://github.com/Iro96/NumBits/blob/main/.github/CONTRIBUTING.md) in this repository.  
> Thanks for helping build NumBits

---

## How To Use

### 1. How to compile

#### Clone the repository

```bah
git clone https://github.com/Iro96/NumBits.git
cd ./NumBits
```

#### Compile your code

```bash
g++ path_to_your_file.cpp -o path_to_your_output -I./include -std=c++2a
path_to_your_output
```

### 2. Example code 
> [!NOTE]
> Please see in `./examples/*.cpp`
