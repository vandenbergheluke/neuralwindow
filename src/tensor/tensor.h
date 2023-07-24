#ifndef TENSOR_H
#define TENSOR_H

#include <errors.h>
#include <datatype.h>
#include <buffer.h>
#include <function.h>

typedef struct function_t function_t;

typedef struct tensor_t
{
    uint64_t id;
    buffer_t *buffer;
    function_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
} tensor_t;

error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient);
void tensor_destroy(tensor_t *tensor);
error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient);
error_t *tensor_addition(tensor_t *x, tensor_t *y, tensor_t *z);
error_t *tensor_matrix_multiplication(tensor_t *x, tensor_t *y, tensor_t *z);
error_t *tensor_copy(tensor_t *source_tensor, tensor_t *destination_tensor);

#endif