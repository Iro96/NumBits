# NumBits

## NumBits Development Plan (v0.1 → v1.0+)

| Version      | Stage                | Focus                          | Description                                                          |
| ------------ | -------------------- | ------------------------------ | -------------------------------------------------------------------- |
| ✅ **v0.1**   | Core MVP             | Array + Basic Ops              | `ndarray<T>`, arithmetic, reductions, random, dot, stats             |
| 🔜 **v0.2**  | Array Manipulation   | Shape ops, broadcasting, views | `reshape`, `transpose`, `expand_dims`, `broadcast_to`, slicing       |
| 🔜 **v0.3**  | Linear Algebra       | Full matrix API                | `matmul`, `inv`, `det`, `eig`, `svd`, `norm`, `trace`                |
| 🔜 **v0.4**  | Advanced Math        | Universal functions            | `sin`, `cos`, `tan`, `log`, `exp`, `pow`, elementwise `ufunc` system |
| 🔜 **v0.5**  | Statistics           | Correlation, covariance        | `corrcoef`, `cov`, `histogram`, `percentile`                         |
| 🔜 **v0.6**  | I/O + Serialization  | File save/load                 | `save`, `load`, `savetxt`, `loadtxt`, binary `.npy` support          |
| 🔜 **v0.7**  | Random 2.0           | Full RNG distributions         | `normal`, `uniform`, `poisson`, `choice`, seeding                    |
| 🔜 **v0.8**  | Backend Acceleration | BLAS / SIMD                    | optional Eigen / OpenBLAS backend, parallel reductions               |
| 🔜 **v0.9**  | GPU Backend          | CUDA / OpenCL / Vulkan         | GPU-enabled ndarray backend                                          |
| 🔜 **v1.0**  | Python API           | `import numbits` via pybind11  | Expose C++ API to Python for hybrid workflows                        |
| 🌟 **v1.1+** | AI / Autograd        | Differentiation engine         | Automatic gradients, neural ops, backpropagation                     |
