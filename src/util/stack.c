#include <stack.h>

error_t *stack_create(stack_t **stack)
{
    CHECK_NULL_ARGUMENT(stack, "stack");

    size_t size = sizeof(stack_t);
    *stack = (stack_t *) malloc(size);
    if (stack == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate stack of size %zu bytes.", size), NULL);
    }
    
    //Initialize
    (*stack)->head = NULL;
    (*stack)->size = 0;

    return NULL;
}

void stack_destroy(stack_t *stack)
{
    if (stack != NULL)
    {
        element_t *element = stack->head;
        while (element != NULL)
        {
            element_t *next = element->next;
            element_destroy(element);
            element = next;
        }
    }
    free(stack);
}

error_t *stack_push(stack_t *stack, void *data)
{
    CHECK_NULL_ARGUMENT(stack, "stack");

    element_t *element;
    error_t *error = element_create(&element, data); 
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create element."), error);
    }

    if (stack->head == NULL)
    {
        stack->head = element;
    }
    else
    {
        element->next = stack->head;
        stack->head = element;
    }
    ++stack->size;

    return NULL;
}

error_t *stack_pop(stack_t *stack, void **data)
{
    CHECK_NULL_ARGUMENT(stack, "stack");
    CHECK_NULL_ARGUMENT(data, "data");

    if (stack->head == NULL)
    {
        return ERROR(ERROR_DESTROY, string_create("failed to pop element from empty stack."), NULL);
    }

    *data = stack->head->data;
    element_t *element = stack->head->next;
    element_destroy(stack->head); 
    stack->head = element;
    stack->size--;

    return NULL;
}
