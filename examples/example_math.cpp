#include <iostream>
#include <cmath>
#include "numbits/ndarray.hpp"
#include "numbits/math_functions.hpp"

using namespace numbits;

int main() {
    std::cout << "=== NumBits Math Functions Example ===\n\n";
    
    // Create ndarray
    ndarray<float> arr({2, 3}, {0.0f, M_PI/4.0f, M_PI/2.0f, M_PI, 2.0f*M_PI, 1.0f});
    
    std::cout << "Original n-D array:\n";
    arr.print();
    std::cout << "\n\n";
    
    // Trigonometric functions
    auto sin_arr = sin(arr);
    std::cout << "sin(arr):\n";
    sin_arr.print();
    std::cout << "\n\n";
    
    auto cos_arr = cos(arr);
    std::cout << "cos(arr):\n";
    cos_arr.print();
    std::cout << "\n\n";
    
    // Exponential and logarithmic
    ndarray<float> exp_arr({1, 4}, {0.0f, 1.0f, 2.0f, -1.0f});
    std::cout << "n-D array for exp/log:\n";
    exp_arr.print();
    std::cout << "\n";
    
    auto exp_result = exp(exp_arr);
    std::cout << "exp(arr):\n";
    exp_result.print();
    std::cout << "\n";
    
    auto log_result = log(exp_result);
    std::cout << "log(exp(arr)) (should equal original):\n";
    log_result.print();
    std::cout << "\n\n";
    
    // Power and sqrt
    ndarray<float> power_arr({2, 2}, {4.0f, 9.0f, 16.0f, 25.0f});
    std::cout << "n-D array:\n";
    power_arr.print();
    std::cout << "\n";
    
    auto sqrt_result = sqrt(power_arr);
    std::cout << "sqrt(arr):\n";
    sqrt_result.print();
    std::cout << "\n";
    
    auto pow_result = pow(power_arr, 0.5f);
    std::cout << "pow(arr, 0.5) (should equal sqrt):\n";
    pow_result.print();
    std::cout << "\n\n";
    
    // Rounding functions
    ndarray<float> round_arr({1, 5}, {1.4f, 1.6f, -1.4f, -1.6f, 2.5f});
    std::cout << "n-D array:\n";
    round_arr.print();
    std::cout << "\n";
    
    auto floor_result = floor(round_arr);
    std::cout << "floor(arr):\n";
    floor_result.print();
    std::cout << "\n";
    
    auto ceil_result = ceil(round_arr);
    std::cout << "ceil(arr):\n";
    ceil_result.print();
    std::cout << "\n";
    
    auto round_result = round(round_arr);
    std::cout << "round(arr):\n";
    round_result.print();
    std::cout << "\n";
    
    return 0;
}

