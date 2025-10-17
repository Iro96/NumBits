#include "../include/numbits/numbits.hpp"
#include <iostream>
#include <memory>

using namespace numbits;

int main() {
    std::cout << "NumBits v0.5 - Statistics Example Extended\n";
    std::cout << "========================================\n\n";

    // --------------------
    // 1D Arrays
    // --------------------
    ndarray<double> A({5}, {1,2,3,4,5});
    ndarray<double> B({5}, {5,4,3,2,1});

    std::cout << "1D Arrays A and B:\n";
    for(auto v : A.data()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "mean(A)      = " << mean(A) << "\n";
    std::cout << "variance(A)  = " << variance(A) << "\n";
    std::cout << "stddev(A)    = " << stddev(A) << "\n";
    std::cout << "cov(A,B)     = " << cov(A,B) << "\n";
    std::cout << "corrcoef(A,B)= " << corrcoef(A,B) << "\n\n";

    // Histogram
    auto [counts, edges] = histogram(A,4);
    std::cout << "Histogram (A, 4 bins):\n";
    for(size_t i=0;i<counts.size();i++)
        std::cout << "  Bin " << i+1 << " [" << edges[i] << ", " << edges[i+1] << "): " << counts[i] << "\n";

    // Percentiles
    std::cout << "Percentiles of A:\n";
    std::cout << "  25th = " << percentile(A,25) << "\n";
    std::cout << "  50th = " << percentile(A,50) << "\n";
    std::cout << "  90th = " << percentile(A,90) << "\n\n";

    // --------------------
    // 2D Array
    // --------------------
    ndarray<double> C({2,5}, {1,2,3,4,5, 2,3,4,6,8});
    std::cout << "2D Array C:\n";
    std::cout << C << "\n";
    std::cout << "Variance(C)  = " << variance(C) << "\n";
    std::cout << "Stddev(C)    = " << stddev(C) << "\n\n";

    std::cout << "Covariance matrix of C:\n" << cov(C) << "\n";
    std::cout << "Correlation matrix of C:\n" << corrcoef(C) << "\n\n";

    // --------------------
    // 3D Array
    // --------------------
    ndarray<double> D({2,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12});
    std::cout << "3D Array D (flattened for mean/variance/stddev):\n";
    std::cout << "mean(D)      = " << mean(D) << "\n";
    std::cout << "variance(D)  = " << variance(D) << "\n";
    std::cout << "stddev(D)    = " << stddev(D) << "\n\n";

    // Histogram 3D
    auto [counts3, edges3] = histogram(D,4);
    std::cout << "Histogram (D, 4 bins):\n";
    for(size_t i=0;i<counts3.size();i++)
        std::cout << "  Bin " << i+1 << " [" << edges3[i] << ", " << edges3[i+1] << "): " << counts3[i] << "\n";
    std::cout << "50th percentile of D = " << percentile(D,50) << "\n\n";

    // --------------------
    // Higher-D Arrays: 4D, 8D, 10D, 14D, 100D
    // --------------------
    // Helper lambda to create ndarray with shared_ptr
    auto make_ndarray = [](const std::vector<size_t>& dims, const std::vector<double>& data){
        auto ptr = std::make_shared<std::vector<double>>(data);
        return ndarray<double>(dims, ptr);
    };

    // 4D
    std::vector<size_t> dims4{2,2,2,3};
    std::vector<double> data4(total_size(dims4));
    for(size_t i=0;i<data4.size();i++) data4[i]=i+1;
    auto D4 = make_ndarray(dims4,data4);
    std::cout << "4D array mean = " << mean(D4) << "\n";

    // 8D
    std::vector<size_t> dims8{2,2,2,2,1,1,1,3};
    std::vector<double> data8(total_size(dims8));
    for(size_t i=0;i<data8.size();i++) data8[i]=i+1;
    auto D8 = make_ndarray(dims8,data8);
    std::cout << "8D array mean = " << mean(D8) << "\n";

    // 10D
    std::vector<size_t> dims10{2,1,1,1,1,1,1,1,2,3};
    std::vector<double> data10(total_size(dims10));
    for(size_t i=0;i<data10.size();i++) data10[i]=i+1;
    auto D10 = make_ndarray(dims10,data10);
    std::cout << "10D array mean = " << mean(D10) << "\n";

    // 14D
    std::vector<size_t> dims14{1,1,1,1,1,1,1,1,1,1,1,1,2,3};
    std::vector<double> data14(total_size(dims14));
    for(size_t i=0;i<data14.size();i++) data14[i]=i+1;
    auto D14 = make_ndarray(dims14,data14);
    std::cout << "14D array mean = " << mean(D14) << "\n";

    // 100D (small demo)
    std::vector<size_t> dims100_small(100,1); // 100 dims of size 1
    std::vector<double> data100_small(total_size(dims100_small),42.0);
    auto D100 = make_ndarray(dims100_small,data100_small);
    std::cout << "100D array mean = " << mean(D100) << "\n";

    std::cout << "\nAll statistical operations executed successfully.\n";
    return 0;
}
