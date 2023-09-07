/**
 * @file view.h
 * @brief Memory view of tensor. 
 */

#ifndef VIEW_H
#define VIEW_H

#include <datatype.h>
#include <errors.h>

/**
 * @brief The maximum number of supported tensor dimensions.
 */
#define MAX_RANK 5

/** 
 *  @brief Defines an interpretation of the underlying storage used to
 *         represent a tensor. 
 * 
 */
typedef struct view_t
{
    uint64_t *shape; /** The dimensions of the tensor. */
    uint64_t rank; /** The rank of the tensor. (The length of shape) */ 
    uint64_t *strides; /** The strides are the jumps necessary to go from one element to the next one in storage along each dimension. (not bytes) */
    uint64_t offset; /** The offset in the underlying storage in terms of number of storage elements. (not bytes) */
} view_t;

nw_error_t *view_create(view_t **view,
                        uint64_t offset,
                        uint64_t rank,
                        const uint64_t *shape,
                        const uint64_t *strides);
void view_destroy(view_t *view);
bool_t is_contiguous(const uint64_t *shape, uint64_t rank, const uint64_t *strides);
nw_error_t *permute(const uint64_t *original_shape,
                    const uint64_t *original_strides,
                    uint64_t *permuted_shape,
                    uint64_t *permuted_strides,
                    const uint64_t *axis,
                    uint64_t length);
nw_error_t *reverse_permute(const uint64_t *axis,
                         uint64_t rank,
                         uint64_t *reverse_axis);
nw_error_t *reduce(const uint64_t *original_shape,
                uint64_t original_rank,
                const uint64_t *original_strides, 
                uint64_t *reduced_shape,
                uint64_t reduced_rank,
                uint64_t *reduced_strides,
                const uint64_t *axis,
                uint64_t rank,
                bool_t keep_dimensions);
nw_error_t *reduce_recover_dimensions(const uint64_t *reduced_shape,
                                      uint64_t reduced_rank, 
                                      const uint64_t *reduced_strides,
                                      uint64_t *recovered_shape, 
                                      uint64_t recovered_rank,
                                      uint64_t *recovered_strides,
                                      const uint64_t *axis,
                                      uint64_t rank);
bool_t shapes_equal(const uint64_t *x_shape, uint64_t x_rank, const uint64_t *y_shape, uint64_t y_rank);
uint64_t shape_size(const uint64_t *shape, uint64_t rank);
nw_error_t *strides_from_shape(uint64_t *strides, const uint64_t *shape, uint64_t rank);
nw_error_t *broadcast_strides(const uint64_t *original_shape,
                           uint64_t original_rank,
                           const uint64_t *original_strides,
                           const uint64_t *broadcasted_shape,
                           uint64_t broadcasted_rank,
                           uint64_t *broadcasted_strides);
nw_error_t *broadcast_shapes(const uint64_t *x_original_shape,
                          uint64_t x_original_rank,
                          const uint64_t *y_original_shape,
                          uint64_t y_original_rank, 
                          uint64_t *broadcasted_shape,
                          uint64_t broadcasted_rank);
nw_error_t *slice_shape(const uint64_t *original_shape,
                     uint64_t original_rank,
                     uint64_t *slice_shape,
                     uint64_t slice_rank,
                     const uint64_t *arguments,
                     uint64_t length);
nw_error_t *slice_offset(const uint64_t *original_strides,
                      uint64_t original_rank,
                      uint64_t *offset,
                      const uint64_t *arguments,
                      uint64_t length);
nw_error_t *reverse_slice(const uint64_t *original_shape,
                       uint64_t original_rank,
                       const uint64_t *arguments,
                       uint64_t length,
                       uint64_t *new_arguments,
                       uint64_t new_length);
nw_error_t *padding(const uint64_t *original_shape,
                 uint64_t original_rank,
                 uint64_t *padding_shape,
                 uint64_t padding_rank,
                 const uint64_t *arguments,
                 uint64_t length);
nw_error_t *reverse_padding(const uint64_t *original_shape,
                         uint64_t original_rank,
                         const uint64_t *arguments,
                         uint64_t length,
                         uint64_t *new_arguments,
                         uint64_t new_length);
nw_error_t *reverse_broadcast_axis(const uint64_t *original_shape,
                                uint64_t original_rank,
                                const uint64_t *broadcasted_shape,
                                uint64_t broadcasted_rank, 
                                uint64_t *axis_keep_dimension,
                                uint64_t *axis_remove_dimension);
nw_error_t *reverse_broadcast_length(const uint64_t *original_shape,
                                  uint64_t original_rank,
                                  const uint64_t *broadcasted_shape,
                                  uint64_t broadcasted_rank, 
                                  uint64_t *length_keep_dimension,
                                  uint64_t *length_remove_dimension);
bool_t is_broadcastable(const uint64_t *original_shape,
                        uint64_t original_rank,
                        const uint64_t *broadcasted_shape,
                        uint64_t broadcasted_rank);
nw_error_t *reduce_compute_buffer_size(const uint64_t *shape,
                                       const uint64_t *strides,
                                       uint64_t rank,
                                       uint64_t n,
                                       const uint64_t *axis,
                                       uint64_t length,
                                       uint64_t *reduced_n);
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif
