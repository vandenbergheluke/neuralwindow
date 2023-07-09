#ifndef TENSOR_H
#define TENSOR_H

#include <datatype.h>
#include <buffer.h>

struct tensor_t;
struct operation_t;
enum operation_type_t;

typedef struct tensor_t
{
    buffer_t *buffer;
    struct operation_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
} tensor_t;

typedef enum operation_type_t
{
    OPERATION_ADDITION
} operation_type_t;

typedef struct operation_t
{
    tensor_t **operands;
    uint32_t number_of_operands;
    operation_type_t type;
} operation_t;

error_t *create_tensor(tensor_t **tensor, runtime_t runtime);
error_t *destroy_tensor(tensor_t *tensor, runtime_t runtime);
error_t *create_operation(operation_t **operation, runtime_t runtime);
error_t *destroy_operation(operation_t *operation, runtime_t runtime);

#endif