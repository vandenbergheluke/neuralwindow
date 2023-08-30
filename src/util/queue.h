#ifndef QUEUE_H
#define QUEUE_H

#include <datatype.h>
#include <element.h>

typedef struct queue_t
{
    element_t *head;
    element_t *tail;
    uint32_t size;
} queue_t;

nw_error_t *queue_create(queue_t **queue);
void queue_destroy(queue_t *queue);
nw_error_t *queue_enqueue(queue_t *queue, void *data);
nw_error_t *queue_dequeue(queue_t *queue, void **data);

#endif