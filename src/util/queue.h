#ifndef QUEUE_H
#define QUEUE_H

#include <datatype.h>
#include <errors.h>

typedef struct element_t
{
    void *data;
    struct element_t *next; 
} element_t;

error_t *element_create(element_t **element, void *data);
void element_destroy(element_t *element);

typedef struct queue_t
{
    element_t *head;
    element_t *tail;
    uint32_t size;
} queue_t;

error_t *queue_create(queue_t **queue);
void queue_destroy(queue_t *queue);
error_t *queue_enqueue(queue_t *queue, void *data);
error_t *queue_dequeue(queue_t *queue, void **data);

#endif