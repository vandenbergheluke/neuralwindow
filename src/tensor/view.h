#ifndef VIEW_H
#define VIEW_H

#include <datatype.h>
#include <errors.h>

#define MAX_RANK 5

typedef struct view_t
{
    uint32_t *shape;
    uint32_t rank; 
    uint32_t *strides;
    uint32_t offset;
} view_t;

nw_error_t *view_create(view_t **view, uint32_t offset, uint32_t rank, const uint32_t *shape, const uint32_t *strides);
void view_destroy(view_t *view);
bool_t is_contiguous(const uint32_t *shape, uint32_t rank, const uint32_t *strides);
nw_error_t *permute(const uint32_t *original_shape,
                 uint32_t original_rank,
                 const uint32_t *original_strides,
                 uint32_t *permuted_shape,
                 uint32_t permuted_rank,
                 uint32_t *permuted_strides,
                 const uint32_t *axis,
                 uint32_t length);
nw_error_t *reverse_permute(const uint32_t *axis,
                         uint32_t rank,
                         uint32_t *reverse_axis);
nw_error_t *reduce(const uint32_t *original_shape,
                uint32_t original_rank,
                const uint32_t *original_strides, 
                uint32_t *reduced_shape,
                uint32_t reduced_rank,
                uint32_t *reduced_strides,
                const uint32_t *axis,
                uint32_t rank,
                bool_t keep_dimensions);
nw_error_t *reduce_recover_dimensions(const uint32_t *original_shape,
                                   uint32_t original_rank,
                                   const uint32_t *original_strides, 
                                   uint32_t *reduced_shape,
                                   uint32_t reduced_rank,
                                   uint32_t *reduced_strides,
                                   const uint32_t *axis,
                                   uint32_t rank);
bool_t shapes_equal(const uint32_t *x_shape, uint32_t x_rank, const uint32_t *y_shape, uint32_t y_rank);
uint32_t shape_size(const uint32_t *shape, uint32_t rank);
nw_error_t *strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank);
nw_error_t *broadcast_strides(const uint32_t *original_shape,
                           uint32_t original_rank,
                           const uint32_t *original_strides,
                           const uint32_t *broadcasted_shape,
                           uint32_t broadcasted_rank,
                           uint32_t *broadcasted_strides);
nw_error_t *broadcast_shapes(const uint32_t *x_original_shape,
                          uint32_t x_original_rank,
                          const uint32_t *y_original_shape,
                          uint32_t y_original_rank, 
                          uint32_t *broadcasted_shape,
                          uint32_t broadcasted_rank);
nw_error_t *slice_shape(const uint32_t *original_shape,
                     uint32_t original_rank,
                     uint32_t *slice_shape,
                     uint32_t slice_rank,
                     const uint32_t *arguments,
                     uint32_t length);
nw_error_t *slice_offset(const uint32_t *original_strides,
                      uint32_t original_rank,
                      uint32_t *offset,
                      const uint32_t *arguments,
                      uint32_t length);
nw_error_t *reverse_slice(const uint32_t *original_shape,
                       uint32_t original_rank,
                       const uint32_t *arguments,
                       uint32_t length,
                       uint32_t *new_arguments,
                       uint32_t new_length);
nw_error_t *padding(const uint32_t *original_shape,
                 uint32_t original_rank,
                 uint32_t *padding_shape,
                 uint32_t padding_rank,
                 const uint32_t *arguments,
                 uint32_t length);
nw_error_t *reverse_padding(const uint32_t *original_shape,
                         uint32_t original_rank,
                         const uint32_t *arguments,
                         uint32_t length,
                         uint32_t *new_arguments,
                         uint32_t new_length);
nw_error_t *reverse_broadcast_axis(const uint32_t *original_shape,
                                uint32_t original_rank,
                                const uint32_t *broadcasted_shape,
                                uint32_t broadcasted_rank, 
                                uint32_t *axis_keep_dimension,
                                uint32_t *axis_remove_dimension);
nw_error_t *reverse_broadcast_length(const uint32_t *original_shape,
                                  uint32_t original_rank,
                                  const uint32_t *broadcasted_shape,
                                  uint32_t broadcasted_rank, 
                                  uint32_t *length_keep_dimension,
                                  uint32_t *length_remove_dimension);
bool_t is_broadcastable(const uint32_t *original_shape,
                        uint32_t original_rank,
                        const uint32_t *broadcasted_shape,
                        uint32_t broadcasted_rank);

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif