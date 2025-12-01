/**
 * @file test_io.cpp
 * @brief Unit tests for file I/O operations (dump, load, tofile, fromfile).
 *
 * Tests the following:
 *   - Binary structured I/O (dump/load with .cb extension)
 *   - Text I/O with various separators (tofile/fromfile)
 *   - Binary I/O without separators
 *   - Type mismatch error handling
 *   - Whitespace flexibility in text parsing
 *
 * @date 2025
 */

#include <iostream>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <vector>
#include "numbits/numbits.hpp"

using namespace numbits;

#define TEST_CASE(name) void name()
#define RUN_TEST(name)  \
    std::cout << "Running " #name "... "; \
    name(); \
    std::cout << "OK\n";

/**
 * @brief Utility function to remove a file if it exists.
 * @param f The filename to remove
 */
static void remove_file(const std::string& f) {
    std::remove(f.c_str());
}

/**
 * @brief Test structured binary dump and load operations.
 */
//   TEST dump / load (structured binary)
TEST_CASE(test_dump_load_structured) {
    ndarray<float> original({2,3}, {1,2,3,4,5,6});

    dump(original, "test_struct.cb");
    auto loaded = load<float>("test_struct.cb");

    assert(original.shape() == loaded.shape());
    assert(original.size() == loaded.size());

    for (size_t i = 0; i < original.size(); i++)
        assert(original.data()[i] == loaded.data()[i]);

    remove_file("test_struct.cb");
}

/**
 * @brief Test text file I/O with newline separator.
 */
//   TEST tofile (text) + fromfile (text)
TEST_CASE(test_text_tofile_fromfile) {
    ndarray<double> arr({5}, {1.5, 2.5, -3.25, 4.0, 10.75});

    tofile(arr, "test_text.txt", "\n");

    auto loaded = fromfile<double>("test_text.txt", "\n");

    assert(loaded.size() == arr.size());
    for (size_t i = 0; i < arr.size(); ++i)
        assert(loaded.data()[i] == arr.data()[i]);

    remove_file("test_text.txt");
}

/**
 * @brief Test binary file I/O with empty separator.
 */
//   TEST tofile (binary) + fromfile (binary)
TEST_CASE(test_binary_tofile_fromfile) {
    ndarray<int> arr({4}, {10, 20, 30, 40});

    // Write binary
    tofile(arr, "test_bin.raw");   // sep = "" → binary

    // Read binary
    auto loaded = fromfile<int>("test_bin.raw");

    assert(loaded.size() == arr.size());
    for (size_t i = 0; i < arr.size(); ++i)
        assert(loaded.data()[i] == arr.data()[i]);

    remove_file("test_bin.raw");
}

/**
 * @brief Test text I/O with custom separators (comma-delimited).
 */
//   TEST arbitrary separator text I/O
TEST_CASE(test_text_sep_comma) {
    ndarray<float> arr({4}, {1.0f, 2.0f, 3.5f, 10.25f});

    tofile(arr, "test_comma.txt", ", ");

    auto loaded = fromfile<float>("test_comma.txt", ", ");

    assert(loaded.size() == arr.size());
    for (size_t i = 0; i < arr.size(); ++i)
        assert(loaded.data()[i] == arr.data()[i]);

    remove_file("test_comma.txt");
}

/**
 * @brief Test that loading with wrong type throws an error.
 */
//   TEST load() type mismatch throws
TEST_CASE(test_type_mismatch) {
    ndarray<double> arr({2}, {1.0, 2.0});
    dump(arr, "type_mismatch.cb");

    bool threw = false;
    try {
        auto wrong = load<float>("type_mismatch.cb"); // wrong type
    }
    catch (...) {
        threw = true;
    }
    assert(threw);

    remove_file("type_mismatch.cb");
}

/**
 * @brief Test text reading with mixed whitespace in separators.
 */
//   TEST fromfile text reading mixed whitespace
TEST_CASE(test_text_whitespace_flexibility) {
    // Create mixed whitespace manually
    {
        std::ofstream f("ws.txt");
        f << "1   2\t3\n4  5\n";
    }

    auto loaded = fromfile<int>("ws.txt", "\n"); // sep="\n" → operator>> behavior

    assert(loaded.size() == 5);
    assert(loaded.data()[0] == 1);
    assert(loaded.data()[1] == 2);
    assert(loaded.data()[2] == 3);
    assert(loaded.data()[3] == 4);
    assert(loaded.data()[4] == 5);

    remove_file("ws.txt");
}

/**
 * @brief Test loading arrays of different types from binary files.
 */
TEST_CASE(test_load_multiple_types) {
    // Test with double
    ndarray<double> arr_double({3}, {1.5, 2.5, 3.5});
    dump(arr_double, "test_double.cb");
    auto loaded = load<double>("test_double.cb");
    assert(loaded.size() == 3);
    assert(std::abs(loaded[0] - 1.5) < 1e-9);
    remove_file("test_double.cb");

    // Test with int
    ndarray<int> arr_int({4}, {10, 20, 30, 40});
    dump(arr_int, "test_int.cb");
    auto loaded_int = load<int>("test_int.cb");
    assert(loaded_int.size() == 4);
    assert(loaded_int[0] == 10);
    remove_file("test_int.cb");
}

/**
 * @brief Test preserving shape through dump and load operations.
 */
TEST_CASE(test_io_preserves_shape) {
    ndarray<float> arr({2, 3, 4}, std::vector<float>(24, 1.5f));
    dump(arr, "test_shape.cb");
    auto loaded = load<float>("test_shape.cb");
    assert((loaded.shape() == Shape{2, 3, 4}));
    remove_file("test_shape.cb");
}

//   Main
int main() {
    std::cout << "=== NumBits IO Tests ===\n\n";

    RUN_TEST(test_dump_load_structured);
    RUN_TEST(test_text_tofile_fromfile);
    RUN_TEST(test_binary_tofile_fromfile);
    RUN_TEST(test_text_sep_comma);
    RUN_TEST(test_type_mismatch);
    RUN_TEST(test_text_whitespace_flexibility);
    RUN_TEST(test_load_multiple_types);
    RUN_TEST(test_io_preserves_shape);

    std::cout << "\nAll tests passed!\n";
    return 0;
}
