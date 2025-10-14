# Start contributing to [NumBits](https://github.com/Iro96/NumBits)

Hello! 👋  
Thank you for your interest in contributing to **NumBits**, a lightweight C++ scientific computing and tensor manipulation library inspired by NumPy and Eigen.  
This guide will walk you through setting up your environment and making your first contribution.

---

## Prerequisites

- [Git](https://git-scm.com/downloads)
- [CMake 3.20+](https://cmake.org/download/)
- [GCC 11+](https://gcc.gnu.org/) or [Clang 13+](https://clang.llvm.org/)
- [Python 3.8+](https://www.python.org/downloads/) *(optional, for testing or benchmarking scripts)*

<details>
<summary>Windows setup</summary>

1. Install [Visual Studio](https://visualstudio.microsoft.com/downloads/) with:
   - **Desktop development with C++**
   - Windows SDK (latest version)
   - CMake component

2. Open **Developer PowerShell** and install Git:

   ```bash
   winget install Git.Git
   ```
   
3. Verify CMake is available:

   ```bash
   cmake --version
   g++ --version
   ```

4. Clone the repository (see in `README.md`).
</details>

<details> 
<summary>Linux setup</summary>

1. Open your terminal.
2. Install build tools and dependencies:
   ```bash
   sudo apt update
   sudo apt install build-essential cmake git
   ```
3. Verify CMake is available:
   ```bash
   cmake --version
   g++ --version
   ```
4. Clone the repository (see in `README.md`).
</details>

---
## Clone and Build NumBits

### 1. Fork the repository

Click the Fork button on the top right corner of the repo page.

### 2. Clone your fork

```bash
git clone https://github.com/<your-github-username>/NumBits.git
cd NumBits
```

### Create your working branch

```bash
git checkout -b <your-branch-name>
```
### 4. Build the library

You can build NumBits using CMake:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run the tests:
```bash
cd build
ctest
```

> [!NOTE]
> If you’re using Visual Studio or CLion, just open the root directory — it’s already a valid CMake project.

---

## Keeping your fork up to date

If you have already cloned the repository, make sure your local copy stays updated:

```bash
git fetch upstream
git checkout main
git pull upstream main --rebase
```

If you are working on a feature branch:

```bash
git checkout <your-branch-name>
git rebase main
```

---

## Development Guidelines

### Code Style

- Follow modern C++17 style.
- Keep all new code header-only if possible (use inline for functions).
- Use snake_case for functions and variables, PascalCase for class names.
- Each feature group (math, linalg, random, etc.) belongs in its own subfolder under `include/numbits/`.

### Documentation

- Use Doxygen-style comments for public APIs:
  ```bash
  /**
  * @brief Performs element-wise addition of two arrays.
  */
  ```
- Every function must have at least a short docstring.

---

## Commit Guidelines

### Before committing

Please verify:
```bash
cmake --build build --config Release
ctest
```

### Commit and push
```bash
git add .
git commit -m "feat(core): add reshape and broadcasting improvements"
git push origin <your-branch-name>
```

If it’s your first time contributing, add the upstream repository:
```bash
git remote add upstream https://github.com/Iro96/NumBits.git
```

---

## Creating a Pull Request

1. Go to your fork on GitHub.
2. Click Compare & pull request.
Ensure:
- Your branch is based on the latest main
- CI tests pass
3. Describe what your contribution adds or fixes.
4. Then click Create pull request

---

## Tips for Contributors

- Use `clang-format` before committing:
  ```bash
  clang-format -i include/**/*.hpp tests/*.cpp
  ```
- Document your changes clearly in PR's description.
- Small, focused PRs are easier to review and merge.
- Include benchmarks or reasoning for any performance-related changes.

---

## Congratulations!

You’ve just contributed to NumBits — a growing C++ scientific computing library!
Thank you for helping make numerical computing in C++ faster, cleaner, and easier.

<p align="center"><b>🚀 Happy Coding! 🚀</b></p>

---

## For Example Usage

Please see in `README.md`
