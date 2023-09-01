#ifndef VIEW_H
#define VIEW_H

#include <datatype.h>
#include <errors.h>

#define MAX_RANK 5

typedef struct view_t
{
    uint64_t *shape;
    uint64_t rank; 
    uint64_t *strides;
    uint64_t offset;
} view_t;

nw_error_t *view_create(view_t **view, uint64_t offset, uint64_t rank, const uint64_t *shape, const uint64_t *strides);
void view_destroy(view_t *view);
bool_t is_contiguous(const uint64_t *shape, uint64_t rank, const uint64_t *strides);
nw_error_t *permute(const uint64_t *original_shape,
                 uint64_t original_rank,
                 const uint64_t *original_strides,
                 uint64_t *permuted_shape,
                 uint64_t permuted_rank,
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
nw_error_t *reduce_recover_dimensions(const uint64_t *original_shape,
                                   uint64_t original_rank,
                                   const uint64_t *original_strides, 
                                   uint64_t *reduced_shape,
                                   uint64_t reduced_rank,
                                   uint64_t *reduced_strides,
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

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif