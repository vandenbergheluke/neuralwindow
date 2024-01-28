#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <errors.h>
#include <tensor.h>

typedef struct softmax_t
{
    int64_t axis;
} softmax_t;

typedef union activation_function_t
{
    softmax_t *softmax;
} activation_function_t;

typedef enum activation_function_type_t
{
    ACTIVATION_RECTIFIED_LINEAR,
    ACTIVATION_SIGMOID,
    ACTIVATION_SOFTMAX,
    ACTIVATION_LOGSOFTMAX,
} activation_function_type_t;

typedef struct activation_t
{
    activation_function_t *activation_function;
    activation_function_type_t activation_function_type;
} activation_t;

nw_error_t *activation_create(activation_t **activation, activation_function_t *activation_function, activation_function_type_t activation_function_type);
void activation_destroy(activation_t *activation);

nw_error_t *activation_function_create(activation_function_t **activation_function, activation_function_type_t activation_function_type, void *type_activation_function);
void activation_function_destroy(activation_function_t *activation_function, activation_function_type_t activation_function_type);

string_t activation_function_type_string(activation_function_type_t activation_function_type);

nw_error_t *softmax_create(softmax_t **softmax, int64_t axis);
void softmax_destroy(softmax_t *softmax);

nw_error_t *rectified_linear_activation_create(activation_t **activation);
nw_error_t *sigmoid_activation_create(activation_t **activation);
nw_error_t *softmax_activation_create(activation_t **activation, int64_t axis);
nw_error_t *logsoftmax_activation_create(activation_t **activation, int64_t axis);

nw_error_t *activation_forward(activation_t *activation, tensor_t *x, tensor_t **y);

#endif