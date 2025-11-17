#include <iostream>
#include <cassert>
#include <algorithm>
#include "numbits/numbits.hpp"

using namespace numbits;

#define TEST_CASE(name) void name()
#define RUN_TEST(name)  \
    std::cout << "Running " #name "... "; \
    name(); \
    std::cout << "OK\n";

// === Test Cases ===

TEST_CASE(test_io_save_load) {
    // Create an ndarray
    ndarray<float> original({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Save to .cb file
    save(original, "test.cb");

    // Load from .cb file
    auto loaded = load<float>("test.cb");

    // Verify shape
    assert(original.shape() == loaded.shape());

    // Verify data matches
    assert(std::equal(
        original.data(),
        original.data() + original.size(),
        loaded.data()
    ));
}

TEST_CASE(test_io_preserves_values) {
    ndarray<float> arr({2, 2}, {3.14f, 2.71f, -1.0f, 0.0f});
    save(arr, "io/values.cb");
    auto loaded = load<float>("io/values.cb");

    // Element-wise comparison
    for (size_t i = 0; i < arr.size(); ++i) {
        assert(arr.data()[i] == loaded.data()[i]);
    }
}

// === Main Runner ===

int main() {
    std::cout << "=== NumBits IO Tests ===\n\n";

    RUN_TEST(test_io_save_load);
    RUN_TEST(test_io_preserves_values);

    std::cout << "\nAll tests passed!\n";
    return 0;
}
