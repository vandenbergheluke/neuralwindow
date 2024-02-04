/**@file buffer.h
 * @brief
 *
 */

#ifndef BUFFER_H
#define BUFFER_H

#include <errors.h>
#include <runtime.h>

typedef struct view_t view_t;

typedef struct storage_t
{
    uint64_t reference_count;
    runtime_t runtime;
    datatype_t datatype;
    int64_t n;
    void *data;
} storage_t;

typedef struct buffer_t
{
    view_t *view;
    storage_t *storage;
} buffer_t;

nw_error_t *buffer_create(buffer_t **buffer, view_t *view, storage_t *storage, bool_t copy);
void buffer_destroy(buffer_t *buffer);
nw_error_t *storage_create(storage_t **storage, runtime_t runtime, datatype_t datatype, int64_t n, void *data, bool_t copy);
void storage_destroy(storage_t *storage);
nw_error_t *buffer_unary(unary_operation_type_t unary_operation_type, buffer_t *x_buffer, buffer_t **y_buffer);
nw_error_t *buffer_binary(binary_operation_type_t operation_type, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t **z_buffer);
nw_error_t *buffer_ternary(ternary_operation_type_t operation_type, buffer_t *w_buffer, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t **z_buffer);
nw_error_t *buffer_reduction(reduction_operation_type_t reduction_operation_type, buffer_t *x, int64_t *axis, int64_t length, buffer_t **result, bool_t keep_dimension);
nw_error_t *buffer_structure(structure_operation_type_t structure_operation_type, buffer_t *x, int64_t *arguments, int64_t length, buffer_t **result);
nw_error_t *buffer_creation(creation_operation_type_t creation_operation_type, buffer_t **buffer, const int64_t *shape, int64_t rank, const int64_t *strides,
                            int64_t offset, const runtime_t runtime, datatype_t datatype, void **arguments, int64_t length, void *data);
#endif
