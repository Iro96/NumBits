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

// Utility: remove file if exists
static void remove_file(const std::string& f) {
    std::remove(f.c_str());
}

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

//   Main
int main() {
    std::cout << "=== NumBits IO Tests ===\n\n";

    RUN_TEST(test_dump_load_structured);
    RUN_TEST(test_text_tofile_fromfile);
    RUN_TEST(test_binary_tofile_fromfile);
    RUN_TEST(test_text_sep_comma);
    RUN_TEST(test_type_mismatch);
    RUN_TEST(test_text_whitespace_flexibility);

    std::cout << "\nAll tests passed!\n";
    return 0;
}
