#ifndef QUEUE_H
#define QUEUE_H

#include <datatype.h>
#include <errors.h>
#include <nw_runtime.h>

typedef struct queue_t
{
    element_t *head;
    element_t *tail;
} queue_t;

typedef struct element_t
{
    void *data;
    struct element_t *next; 
} element_t;

error_t *create_queue(queue_t **queue);
error_t *destroy_queue(queue_t *queue);
error_t *create_element(element_t **element, void *data);
error_t *destroy_element(element_t *element);
error_t *enqueue(queue_t *queue, void *data);
error_t *dequeue(queue_t *queue, void **data);

#endif