/**
 * @file io.hpp
 * @brief File I/O operations for saving and loading arrays.
 *
 * Provides multiple I/O formats:
 *   - Binary structured I/O (dump/load): Stores shape, type, and data
 *   - Text I/O (tofile/fromfile): Human-readable text with custom separators
 *   - Raw binary I/O: Stores only data without metadata
 *
 * @namespace numbits
 */

#pragma once

#include "ndarray.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

namespace numbits {

/**
 * @brief Ensure a filename ends with the `.cb` extension.
 *
 * Used internally by dump() and load() to enforce a consistent
 * binary container file extension. If the filename already ends
 * with `.cb` (case-insensitive), it is returned unchanged; otherwise
 * `.cb` is appended.
 *
 * @param filename The input filename.
 * @return A filename guaranteed to end with `.cb`.
 */
inline std::string ensure_cb_extension(const std::string& filename) {
    if (filename.size() >= 3) {
        std::string ext = filename.substr(filename.size() - 3);
        for (auto &ch : ext) ch = static_cast<char>(std::tolower(ch));
        if (ext == ".cb") return filename;
    }
    return filename + ".cb";
}


/**
 * @brief Save an ndarray to a file in text or binary format, similar to NumPy's `tofile`.
 *
 * When `sep` is an empty string (default), the file is written in binary mode
 * as a raw contiguous dump of the array’s data buffer (no metadata).
 * When `sep` is non-empty, the file is written in text mode with each element
 * written sequentially separated by `sep`.
 *
 * @tparam T Element type stored in the ndarray.
 * @param arr The ndarray to serialize.
 * @param filename Path to the output file.
 * @param sep Separator string.  
 *        - `""` → binary mode  
 *        - non-empty → text mode with separator between values
 *
 * @throws std::runtime_error if the file cannot be opened or written.
 */
template<typename T>
void tofile(const ndarray<T>& arr,
            const std::string& filename,
            const std::string& sep = "")
{
    bool binary = sep.empty();

    if (binary) {
        // ---- binary write identical to NumPy tofile(..., sep='') ----
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);

        file.write(reinterpret_cast<const char*>(arr.data()),
                   arr.size() * sizeof(T));

        if (!file) throw std::runtime_error("Error writing binary tofile: " + filename);
    }
    else {
        // ---- text mode: one element written at a time ----
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);

        size_t n = arr.size();
        for (size_t i = 0; i < n; ++i) {
            file << arr.data()[i];
            if (i + 1 < n) file << sep;
        }
        if (!file) throw std::runtime_error("Error writing text tofile: " + filename);
    }
}


/**
 * @brief Load an ndarray from a text or binary file, similar to NumPy's `fromfile`.
 *
 * When `sep` is empty, the file is interpreted as a raw binary buffer containing
 * contiguous elements of type `T`. File size must be an exact multiple of `sizeof(T)`.
 *
 * When `sep` is non-empty, text mode is used. Values are parsed sequentially:
 *  - If `sep == "\\n"` → tokenized by whitespace via standard input streaming.
 *  - Otherwise → the entire file is read and split using the custom `sep`.
 *
 * The resulting array is always 1-dimensional.
 *
 * @tparam T Element type to read.
 * @param filename Path to the input file.
 * @param sep Separator string.  
 *        - `""` → binary mode  
 *        - non-empty → text mode
 *
 * @return ndarray<T> containing the loaded values.
 *
 * @throws std::runtime_error if the file cannot be opened or parsing fails.
 */
template<typename T>
ndarray<T> fromfile(const std::string& filename,
                    const std::string& sep = "")
{
    bool binary = sep.empty();

    if (binary) {
        // ---- binary mode ----
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + filename);

        // Determine file size
        file.seekg(0, std::ios::end);
        std::streamsize bytes = file.tellg();
        file.seekg(0, std::ios::beg);

        if (bytes % sizeof(T) != 0)
            throw std::runtime_error("Binary fromfile size mismatch");

        size_t count = bytes / sizeof(T);
        ndarray<T> arr({count});

        file.read(reinterpret_cast<char*>(arr.data()), bytes);
        if (!file) throw std::runtime_error("Error reading binary fromfile");

        return arr;
    }
    else {
        // ---- text mode ----
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + filename);

        std::vector<T> values;
        std::string token;

        if (sep == "\n") {
            T v;
            while (file >> v) values.push_back(v);
        } else {
            // general separator: read whole file then split
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string data = buffer.str();

            size_t start = 0;
            while (true) {
                size_t pos = data.find(sep, start);
                std::string piece = (pos == std::string::npos)
                                      ? data.substr(start)
                                      : data.substr(start, pos - start);
                if (!piece.empty()) {
                    std::stringstream ss(piece);
                    T v;
                    ss >> v;
                    values.push_back(v);
                }
                if (pos == std::string::npos) break;
                start = pos + sep.size();
            }
        }

        return ndarray<T>(values);
    }
}


/**
 * @brief Dump an ndarray to a structured binary file (similar to NumPy `.npy`/dump).
 *
 * The file format contains:
 *  1. `DType` enum describing the stored type  
 *  2. `ndim` (size_t)  
 *  3. Each dimension size (size_t)  
 *  4. Total element count (size_t)  
 *  5. Raw contiguous data buffer
 *
 * File extension `.cb` is enforced automatically.
 *
 * @tparam T Element type.
 * @param arr Array to serialize.
 * @param filename Base filename (extension appended if needed).
 *
 * @throws std::runtime_error if writing fails.
 */
template<typename T>
void dump(const ndarray<T>& arr, const std::string& filename)
{
    std::string full_filename = ensure_cb_extension(filename);
    std::ofstream file(full_filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for writing: " + full_filename);

    // Write dtype
    DType dtype = dtype_from_type<T>();
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(DType));

    // Write shape
    size_t ndim = arr.shape().size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(size_t));
    for (size_t dim : arr.shape()) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
    }

    // Write size
    size_t size = arr.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    // Write raw binary payload
    file.write(reinterpret_cast<const char*>(arr.data()), size * sizeof(T));

    if (!file) throw std::runtime_error("Error writing dump file: " + full_filename);
}


/**
 * @brief Load an ndarray from a structured binary `.cb` file written by dump().
 *
 * The loader verifies:
 *  - Stored `DType` matches `T`
 *  - Stored shape dimensions multiply to the stored element count
 *
 * On success, the ndarray is allocated with the correct shape and filled
 * with binary data from the file.
 *
 * @tparam T Expected element type.
 * @param filename Path to the `.cb` binary file.
 *
 * @return ndarray<T> restored from disk.
 *
 * @throws std::runtime_error on type mismatch, shape inconsistency, or I/O failure.
 */
template<typename T>
ndarray<T> load(const std::string& filename)
{
    std::string full_filename = ensure_cb_extension(filename);
    std::ifstream file(full_filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + full_filename);

    // Read dtype
    DType dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(DType));
    if (dtype != dtype_from_type<T>())
        throw std::runtime_error("Type mismatch: " + full_filename);

    // Read shape
    size_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(size_t));
    Shape shape(ndim);
    for (size_t i = 0; i < ndim; ++i)
        file.read(reinterpret_cast<char*>(&shape[i]), sizeof(size_t));

    // Read size
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    size_t expected = compute_size(shape);
    if (size != expected)
        throw std::runtime_error("Shape-size mismatch in: " + full_filename);

    // Allocate
    ndarray<T> arr(shape);

    // Read raw data
    file.read(reinterpret_cast<char*>(arr.data()), size * sizeof(T));
    if (!file) throw std::runtime_error("Error reading dump: " + full_filename);

    return arr;
}

} // namespace numbits
