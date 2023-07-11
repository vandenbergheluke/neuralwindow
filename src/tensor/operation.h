
#ifndef OPERATION_H
#define OPERATION_H

#include <datatype.h>
#include <errors.h>

struct tensor_t;
enum operation_type_t;
struct operation_t;

#include <tensor.h>

typedef enum operation_type_t
{
    NULL_OPERATION,
    OPERATION_ADDITION
} operation_type_t;

typedef struct operation_t
{
    struct tensor_t **operands;
    operation_type_t operation_type;
} operation_t;

error_t *create_operation(operation_t **operation, runtime_t runtime);
error_t *destroy_operation(operation_t *operation, runtime_t runtime);
uint32_t number_of_operands(operation_type_t operation_type);

#endif