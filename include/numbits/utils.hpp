#pragma once

#include "types.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <string>

namespace numbits {

inline size_t compute_size(const Shape& shape) {
    if (shape.empty()) return 1;
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

inline Strides compute_strides(const Shape& shape) {
    if (shape.empty()) return {};
    Strides strides(shape.size());
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

inline size_t flatten_index(const std::vector<size_t>& indices, const Strides& strides) {
    size_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

inline std::vector<size_t> unravel_index(size_t flat_idx, const Shape& shape, const Strides& strides) {
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        indices[i] = flat_idx / strides[i];
        flat_idx %= strides[i];
    }
    return indices;
}

inline Shape broadcast_shapes(const Shape& shape1, const Shape& shape2) {
    size_t ndim = std::max(shape1.size(), shape2.size());
    Shape result(ndim);
    
    for (size_t i = 0; i < ndim; ++i) {
        size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::runtime_error("Cannot broadcast shapes");
        }
        
        result[ndim - 1 - i] = std::max(dim1, dim2);
    }
    
    return result;
}

inline bool can_broadcast(const Shape& shape1, const Shape& shape2) {
    try {
        broadcast_shapes(shape1, shape2);
        return true;
    } catch (...) {
        return false;
    }
}

inline std::string shape_to_string(const Shape& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    if (shape.size() == 1) oss << ",";
    oss << ")";
    return oss.str();
}

} // namespace numbits

