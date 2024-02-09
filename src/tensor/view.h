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

int64_t array_product(const int64_t *array, int64_t length);
nw_error_t *view_copy(const view_t *source_view, view_t **destination_view);
nw_error_t *view_create(view_t **view, int64_t offset, int64_t rank, const int64_t *shape, const int64_t *strides);
void view_destroy(view_t *view);
int64_t dimension_to_index(int64_t dimension, int64_t rank);
nw_error_t *strides_from_shape(int64_t *strides, const int64_t *shape, int64_t rank);
nw_error_t *view_permute(const view_t *original_view, view_t **permuted_view, const int64_t *axis, int64_t length);
nw_error_t *view_reduce(const view_t *original_view, view_t **reduced_view, const int64_t *axis, int64_t length, bool_t keep_dimensions);
nw_error_t *view_recover_dimensions(const view_t *reduced_view, view_t **recovered_view, const int64_t *axis, int64_t length);
nw_error_t *view_physical_size(const view_t *view, int64_t *size);
nw_error_t *view_logical_size(const view_t *view, int64_t *size);
bool_t view_shapes_equal(const view_t *view_a, const view_t *view_b);
bool_t view_has_shape(const view_t *view, const int64_t *shape, int64_t rank);
nw_error_t *view_is_contiguous(const view_t *view, bool_t *is_contiguous);
nw_error_t *view_expand(const view_t *original_view, view_t **expanded_view, const int64_t *shape, int64_t rank);
nw_error_t *view_broadcast(const view_t *view_a, const view_t *view_b, int64_t **shape, int64_t *rank);
nw_error_t *view_broadcast_matrix_multiplication(const view_t *view_a, const view_t *view_b, int64_t **shape_a, int64_t **shape_b, int64_t *rank);
nw_error_t *view_matrix_multiplication(const view_t *view_a, const view_t *view_b, view_t **view_c);
nw_error_t *view_reduce_axis(const view_t *original_view, 
                             const int64_t *broadcasted_shape, int64_t broadcasted_rank,
                             int64_t **axis_keep_dimension, int64_t *length_keep_dimension,
                             int64_t **axis_remove_dimension, int64_t *length_remove_dimension);
nw_error_t *view_slice(const view_t *original_view, view_t **sliced_view, const int64_t *arguments, int64_t length);
nw_error_t *view_slice_padding_arguments(const view_t *original_view, const int64_t *slice_arguments, int64_t length, int64_t **padding_arguments);
nw_error_t *view_padding(const view_t *original_view, view_t **padding_view, const int64_t *arguments, int64_t length);
nw_error_t *view_padding_slice_arguments(const view_t *original_view, const int64_t *padding_arguments, int64_t length, int64_t **slice_arguments);
#endif
