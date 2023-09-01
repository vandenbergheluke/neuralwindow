/**@file stack.h
 * @brief Provides stack type and its utilities.
 *
 */

#ifndef STACK_H
#define STACK_H

#include <element.h>
#include <datatype.h>
#include <errors.h>

typedef struct stack_t
{
    element_t *head;
    uint64_t size;
} stack_t;

nw_error_t *stack_create(stack_t **stack);
void stack_destroy(stack_t *stack);
nw_error_t *stack_push(stack_t *stack, void *data);
nw_error_t *stack_pop(stack_t *stack, void **data);

#endif
