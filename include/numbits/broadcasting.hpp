/**
 * @file broadcasting.hpp
 * @brief Broadcasting utilities for NumPy-like array dimension expansion.
 *
 * Provides functionality to broadcast arrays to compatible shapes for
 * element-wise operations. Follows NumPy broadcasting rules:
 *   - Arrays are aligned from the right.
 *   - Dimensions with size 1 can be stretched to match the other array.
 *   - Missing dimensions are treated as size 1.
 *
 * @namespace numbits
 */

#pragma once

#include "ndarray.hpp"
#include "utils.hpp"
#include <vector>

namespace numbits {

/**
 * @class BroadcastIterator
 * @brief Iterator that produces broadcasted values from a given ndarray.
 *
 * This class simulates iteration over an array after broadcasting it
 * to a target shape, without actually allocating a new array.
 *
 * It expands the original ndarray shape following NumPy broadcasting rules,
 * computes appropriate stride transformations, and allows sequential access
 * to elements as if the data were broadcasted in memory.
 *
 * @tparam T Element type of the ndarray.
 */
template<typename T>
class BroadcastIterator {
public:

    /**
     * @brief Construct a BroadcastIterator for an ndarray and target shape.
     *
     * The constructor generates:
     *   - expanded_shape_: original shape aligned to the right and padded with 1s.
     *   - expanded_strides_: strides for the expanded shape.
     *   - internal index counters for sweeping through the broadcasted space.
     *
     * @param arr The ndarray to broadcast.
     * @param target_shape The shape to broadcast to (must be broadcast-compatible).
     */
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

    /**
     * @brief Retrieve the value of the original ndarray at the broadcasted position.
     *
     * This computes the corresponding index in the original ndarray by:
     *   - Using current_index_ to determine the implied multi-index.
     *   - Applying broadcasting rules: if the original dimension is 1, index remains 0.
     *   - Computing flat index using expanded_strides_.
     *
     * @return T The value from the ndarray at the broadcasted coordinate.
     */
    T get_value() const {
        size_t ndarray_index = 0;
        for (size_t i = 0; i < target_shape_.size(); ++i) {
            size_t dim_index = current_index_[i];
            if (expanded_shape_[i] == 1) {
                dim_index = 0;
            }
            ndarray_index += dim_index * expanded_strides_[i];
        }
        return ndarray_.data()[ndarray_index];
    }

    /**
     * @brief Increment iterator to the next broadcasted element.
     *
     * This behaves like a multi-dimensional counter that wraps back
     * when a dimension is exhausted, similar to iterating through a
     * lexicographic index sequence.
     */
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

    /**
     * @brief Check whether the iterator has reached the end.
     *
     * @return true If all broadcasted elements have been iterated.
     * @return false Otherwise.
     */
    bool is_end() const {
        return flat_index_ >= compute_size(target_shape_);
    }

    /**
     * @brief Get linear index of current iteration.
     *
     * @return size_t Flat index in the broadcasted result space.
     */
    size_t flat_index() const { return flat_index_; }

private:
    const ndarray<T>& ndarray_;     ///< Reference to source ndarray.
    Shape target_shape_;            ///< Full broadcasted shape.
    Strides target_strides_;        ///< Strides corresponding to target shape.
    Strides ndarray_strides_;       ///< Original ndarray strides.
    Shape expanded_shape_;          ///< ndarray shape aligned to target shape.
    Strides expanded_strides_;      ///< Strides for the expanded shape.
    std::vector<size_t> current_index_; ///< Multi-index cursor.
    size_t flat_index_;             ///< Current flat index.
};

/**
 * @brief Broadcast an ndarray to the desired target shape.
 *
 * Allocates a new ndarray with broadcasted dimensions and fills it
 * using a BroadcastIterator. The broadcasting follows NumPy rules
 * for dimension expansion and alignment.
 *
 * @tparam T Element type of the ndarray.
 * @param arr The array to broadcast.
 * @param target_shape The desired output shape.
 *
 * @return ndarray<T> A new ndarray with the broadcasted shape.
 *
 * @throws std::runtime_error If the shapes are not broadcast-compatible.
 */
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
