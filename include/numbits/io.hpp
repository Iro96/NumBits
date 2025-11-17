#pragma once

#include "ndarray.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

namespace numbits {

inline std::string ensure_cb_extension(const std::string& filename) {
    if (filename.size() >= 3) {
        std::string ext = filename.substr(filename.size() - 3);
        // Convert to lowercase for comparison
        for (auto &ch : ext) ch = static_cast<char>(std::tolower(ch));
        if (ext == ".cb") {
            return filename; // already has .cb or .CB etc.
        }
    }
    return filename + ".cb";
}

template<typename T>
void save(const ndarray<T>& arr, const std::string& filename) {
    std::string full_filename = ensure_cb_extension(filename);
    std::ofstream file(full_filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + full_filename);
    }

    // Write DType
    DType dtype = dtype_from_type<T>();
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(DType));

    // Write Shape
    size_t ndim = arr.shape().size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(size_t));
    for (size_t dim : arr.shape()) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
    }

    // Write size for verification
    size_t size = arr.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    // Write data
    file.write(reinterpret_cast<const char*>(arr.data()), size * sizeof(T));

    if (!file) {
        throw std::runtime_error("Error writing to file: " + full_filename);
    }
}

template<typename T>
ndarray<T> load(const std::string& filename) {
    std::string full_filename = ensure_cb_extension(filename);
    std::ifstream file(full_filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + full_filename);
    }

    // Read DType
    DType dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(DType));
    if (dtype != dtype_from_type<T>()) {
        throw std::runtime_error("Type mismatch in file: " + full_filename);
    }

    // Read Shape
    size_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(size_t));
    Shape shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&shape[i]), sizeof(size_t));
    }

    // Read and verify size
    size_t file_size;
    file.read(reinterpret_cast<char*>(&file_size), sizeof(size_t));
    size_t expected_size = compute_size(shape);
    if (file_size != expected_size) {
        throw std::runtime_error("Size mismatch in file: " + full_filename);
    }

    // Create ndarray
    ndarray<T> arr(shape);

    // Read data
    file.read(reinterpret_cast<char*>(arr.data()), expected_size * sizeof(T));

    if (!file) {
        throw std::runtime_error("Error reading from file: " + full_filename);
    }

    return arr;
}

} // namespace numbits