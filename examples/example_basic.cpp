#include <iostream>
#include "numbits/ndarray.hpp"
#include "numbits/operations.hpp"

using namespace numbits;

int main() {
    std::cout << "=== NumBits Basic Example ===\n\n";
    
    // Create ndarrays
    ndarray<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ndarray<float> b({2, 3}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    
    std::cout << "n-D array a:\n";
    a.print();
    std::cout << "\n\n";
    
    std::cout << "n-D array b:\n";
    b.print();
    std::cout << "\n\n";
    
    // Basic operations
    auto c = a + b;
    std::cout << "a + b:\n";
    c.print();
    std::cout << "\n\n";
    
    auto d = a * 2.0f;
    std::cout << "a * 2:\n";
    d.print();
    std::cout << "\n\n";
    
    // Scalar operations
    auto e = a + 10.0f;
    std::cout << "a + 10:\n";
    e.print();
    std::cout << "\n\n";
    
    // Create zeros and ones
    auto zeros = ndarray<float>::zeros({3, 4});
    std::cout << "Zeros (3x4):\n";
    zeros.print();
    std::cout << "\n\n";
    
    auto ones = ndarray<float>::ones({2, 2});
    std::cout << "Ones (2x2):\n";
    ones.print();
    std::cout << "\n\n";
    
    // Reshape
    auto flattened = a.flatten();
    std::cout << "Flattened a:\n";
    flattened.print();
    std::cout << "\n\n";
    
    auto reshaped = flattened.reshape({3, 2});
    std::cout << "Reshaped to (3x2):\n";
    reshaped.print();
    std::cout << "\n\n";
    
    // Sum and mean
    std::cout << "Sum of a: " << sum(a) << "\n";
    std::cout << "Mean of a: " << mean(a) << "\n";
    std::cout << "Min of a: " << min(a) << "\n";
    std::cout << "Max of a: " << max(a) << "\n";
    
    return 0;
}

