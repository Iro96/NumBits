# Building NumBits

## Quick Start

### Windows (PowerShell)

```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the library
cmake --build . --config Release

# Build examples
cmake --build . --config Release --target example_basic
cmake --build . --config Release --target example_linear_algebra
cmake --build . --config Release --target example_math
cmake --build . --config Release --target example_broadcasting

# Run examples
.\examples\Release\example_basic.exe
.\examples\Release\example_linear_algebra.exe
.\examples\Release\example_math.exe
.\examples\Release\example_broadcasting.exe
```

### Linux/macOS

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the library and examples
cmake --build .

# Run examples
./examples/example_basic
./examples/example_linear_algebra
./examples/example_math
./examples/example_broadcasting
```

## Using NumBits in Your Project

### CMake Integration

```cmake
# Add NumBits as a subdirectory or install it first
add_subdirectory(path/to/NumBits)

# Link against NumBits
target_link_libraries(your_target numbits)

# Include headers
target_include_directories(your_target PRIVATE path/to/NumBits/include)
```

### Direct Include

```cpp
#include "numbits/numbits.hpp"  // Includes everything
// or
#include "numbits/array.hpp"
#include "numbits/operations.hpp"
// ... etc
```

## Building Tests

Tests require Catch2. Install Catch2 first, then:

```bash
cmake .. -DNUMBITS_BUILD_TESTS=ON
cmake --build .
ctest
```

