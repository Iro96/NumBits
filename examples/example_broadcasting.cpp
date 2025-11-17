#include <iostream>
#include "numbits/ndarray.hpp"
#include "numbits/operations.hpp"
#include "numbits/broadcasting.hpp"

using namespace numbits;

int main() {
    std::cout << "=== NumBits Broadcasting Example ===\n\n";
    
    // Create ndarrays of different shapes
    ndarray<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ndarray<float> b({3}, {10.0f, 20.0f, 30.0f});
    
    std::cout << "n-D array a (2x3):\n";
    a.print();
    std::cout << "\n\n";
    
    std::cout << "n-D array b (3):\n";
    b.print();
    std::cout << "\n\n";
    
    // Broadcasting: a + b
    auto c = a + b;
    std::cout << "a + b (broadcasted):\n";
    c.print();
    std::cout << "\n\n";
    
    // Scalar broadcasting
    ndarray<float> scalar_arr({1}, {5.0f});
    auto d = a + scalar_arr;
    std::cout << "a + scalar (5):\n";
    d.print();
    std::cout << "\n\n";
    
    // Different broadcasting scenario
    ndarray<float> row({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    ndarray<float> col({4, 1}, {1.0f, 10.0f, 100.0f, 1000.0f});
    
    std::cout << "Row vector (1x4):\n";
    row.print();
    std::cout << "\n\n";
    
    std::cout << "Column vector (4x1):\n";
    col.print();
    std::cout << "\n\n";
    
    auto broadcasted = row + col;
    std::cout << "Row + Column (broadcasted to 4x4):\n";
    broadcasted.print();
    std::cout << "\n\n";
    
    // Element-wise multiplication with broadcasting
    ndarray<float> matrix({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    ndarray<float> vec({3}, {2.0f, 3.0f, 4.0f});
    
    std::cout << "Matrix (3x3):\n";
    matrix.print();
    std::cout << "\n\n";
    
    std::cout << "Vector (3):\n";
    vec.print();
    std::cout << "\n\n";
    
    auto multiplied = matrix * vec;
    std::cout << "Matrix * Vector (broadcasted):\n";
    multiplied.print();
    std::cout << "\n";
    
    return 0;
}

