#pragma once

#include "ndarray.hpp"
#include "utils.hpp"
#include <vector>

namespace numbits {

template<typename T>
class BroadcastIterator {
public:
    BroadcastIterator(const ndarray<T>& arr, const Shape& target_shape)
        : ndarray_(arr), target_shape_(target_shape), 
          target_strides_(compute_strides(target_shape)),
          ndarray_strides_(arr.strides()),
          current_index_(target_shape.size(), 0),
          flat_index_(0) {
        // Expand ndarray shape to match target dimensions
        expanded_shape_.resize(target_shape.size(), 1);
        size_t offset = target_shape.size() - arr.ndim();
        for (size_t i = 0; i < arr.ndim(); ++i) {
            expanded_shape_[offset + i] = arr.shape()[i];
        }
        expanded_strides_ = compute_strides(expanded_shape_);
    }

    T get_value() const {
        // Calculate the index in the original ndarray
        size_t ndarray_index = 0;
        for (size_t i = 0; i < target_shape_.size(); ++i) {
            size_t dim_index = current_index_[i];
            // If this dimension was 1 in the original, use index 0
            if (expanded_shape_[i] == 1) {
                dim_index = 0;
            }
            ndarray_index += dim_index * expanded_strides_[i];
        }
        return ndarray_.data()[ndarray_index];
    }

    void increment() {
        for (int i = static_cast<int>(target_shape_.size()) - 1; i >= 0; --i) {
            current_index_[i]++;
            if (current_index_[i] < target_shape_[i]) {
                break;
            }
            current_index_[i] = 0;
        }
        flat_index_++;
    }

    bool is_end() const {
        return flat_index_ >= compute_size(target_shape_);
    }

    size_t flat_index() const { return flat_index_; }

private:
    const ndarray<T>& ndarray_;
    Shape target_shape_;
    Strides target_strides_;
    Strides ndarray_strides_;
    Shape expanded_shape_;
    Strides expanded_strides_;
    std::vector<size_t> current_index_;
    size_t flat_index_;
};

template<typename T>
ndarray<T> broadcast_to(const ndarray<T>& arr, const Shape& target_shape) {
    Shape broadcasted_shape = broadcast_shapes(arr.shape(), target_shape);
    ndarray<T> result(broadcasted_shape);
    
    BroadcastIterator<T> it(arr, broadcasted_shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = it.get_value();
        it.increment();
    }
    
    return result;
}

} // namespace numbits

