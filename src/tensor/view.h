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
    int64_t *shape; /** The dimensions of the tensor. */
    int64_t rank; /** The rank of the tensor. (The length of shape) */ 
    int64_t *strides; /** The strides are the jumps necessary to go from one element to the next one in storage along each dimension. (not bytes) */
    int64_t offset; /** The offset in the underlying storage in terms of number of storage elements. (not bytes) */
} view_t;

nw_error_t *view_create(view_t **view,
                        int64_t offset,
                        int64_t rank,
                        const int64_t *shape,
                        const int64_t *strides);
void view_destroy(view_t *view);
bool_t is_contiguous(const int64_t *shape,
                     int64_t rank,
                     const int64_t *strides,
                     int64_t offset);
nw_error_t *permute(const int64_t *original_shape,
                    const int64_t *original_strides,
                    int64_t *permuted_shape,
                    int64_t *permuted_strides,
                    const int64_t *axis,
                    int64_t length);
nw_error_t *reverse_permute(const int64_t *axis,
                         int64_t rank,
                         int64_t *reverse_axis);
nw_error_t *reduce(const int64_t *original_shape,
                int64_t original_rank,
                const int64_t *original_strides, 
                int64_t *reduced_shape,
                int64_t reduced_rank,
                int64_t *reduced_strides,
                const int64_t *axis,
                int64_t rank,
                bool_t keep_dimensions);
nw_error_t *reduce_recover_dimensions(const int64_t *reduced_shape,
                                      int64_t reduced_rank, 
                                      const int64_t *reduced_strides,
                                      int64_t *recovered_shape, 
                                      int64_t recovered_rank,
                                      int64_t *recovered_strides,
                                      const int64_t *axis,
                                      int64_t rank);
bool_t shapes_equal(const int64_t *x_shape, int64_t x_rank, const int64_t *y_shape, int64_t y_rank);
int64_t shape_size(const int64_t *shape, int64_t rank);
nw_error_t *strides_from_shape(int64_t *strides, const int64_t *shape, int64_t rank);
nw_error_t *broadcast_strides(const int64_t *original_shape,
                           int64_t original_rank,
                           const int64_t *original_strides,
                           const int64_t *broadcasted_shape,
                           int64_t broadcasted_rank,
                           int64_t *broadcasted_strides);
nw_error_t *broadcast_shapes(const int64_t *x_original_shape,
                          int64_t x_original_rank,
                          const int64_t *y_original_shape,
                          int64_t y_original_rank, 
                          int64_t *broadcasted_shape,
                          int64_t broadcasted_rank);
nw_error_t *matrix_multiplication_broadcast_shapes(const int64_t *x_original_shape,
                                                   int64_t x_original_rank,
                                                   const int64_t *y_original_shape,
                                                   int64_t y_original_rank, 
                                                   int64_t *x_broadcasted_shape,
                                                   int64_t *y_broadcasted_shape,
                                                   int64_t broadcasted_rank);
nw_error_t *matrix_multiplication_shape(int64_t *x_shape, int64_t *y_shape, int64_t *z_shape, int64_t rank);
nw_error_t *slice_shape(const int64_t *original_shape,
                     int64_t original_rank,
                     int64_t *slice_shape,
                     int64_t slice_rank,
                     const int64_t *arguments,
                     int64_t length);
nw_error_t *slice_offset(const int64_t *original_strides,
                      int64_t original_rank,
                      int64_t *offset,
                      const int64_t *arguments,
                      int64_t length);
nw_error_t *reverse_slice(const int64_t *original_shape,
                       int64_t original_rank,
                       const int64_t *arguments,
                       int64_t length,
                       int64_t *new_arguments,
                       int64_t new_length);
nw_error_t *padding(const int64_t *original_shape,
                 int64_t original_rank,
                 int64_t *padding_shape,
                 int64_t padding_rank,
                 const int64_t *arguments,
                 int64_t length);
nw_error_t *reverse_padding(const int64_t *original_shape,
                         int64_t original_rank,
                         const int64_t *arguments,
                         int64_t length,
                         int64_t *new_arguments,
                         int64_t new_length);
nw_error_t *reduce_axis(const int64_t *original_shape,
                                int64_t original_rank,
                                const int64_t *broadcasted_shape,
                                int64_t broadcasted_rank, 
                                int64_t *axis_keep_dimension,
                                int64_t *axis_remove_dimension);
nw_error_t *reduce_axis_length(const int64_t *original_shape,
                                  int64_t original_rank,
                                  const int64_t *broadcasted_shape,
                                  int64_t broadcasted_rank, 
                                  int64_t *length_keep_dimension,
                                  int64_t *length_remove_dimension);
bool_t is_broadcastable(const int64_t *original_shape,
                        int64_t original_rank,
                        const int64_t *broadcasted_shape,
                        int64_t broadcasted_rank);
nw_error_t *n_from_shape_and_strides(const int64_t *shape, 
                                     const int64_t *strides,
                                     int64_t rank,
                                     int64_t *n);
bool_t is_valid_reshape(const int64_t *original_shape, int64_t original_rank,
                        const int64_t *new_shape, int64_t new_rank);
nw_error_t *view_copy(const view_t *source_view, view_t **destination_view);

#endif
