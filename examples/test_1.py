#include <numbits/numbits.hpp>
using namespace nb;

int main() {
    ndarray<double> a({5}, 1.5);
    ndarray<double> b({5}, 2.0);

    auto c = add(a, b);
    auto d = multiply(c, b);

    c.print();
    d.print();

    std::cout << "sum(d): " << sum(d) << std::endl;
    std::cout << "mean(d): " << mean(d) << std::endl;

    auto r = rand<double>({5});
    r.print();

    std::cout << "dot(a,b): " << dot(a,b) << std::endl;
    std::cout << "stddev(r): " << stddev(r) << std::endl;
}
