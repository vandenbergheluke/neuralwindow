#ifndef OPERATION_H
#define OPERATION_H

typedef union operation_t
{
    binary_operation_t binary_operation;
} operation_t;

typedef enum binary_operation_t
{
    ADDITION_OPERATION
} binary_operation_t;

#endif
