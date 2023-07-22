#include <queue.h>

error_t *element_create(element_t **element, void *data)
{
    CHECK_NULL_ARGUMENT(element, "element");

    size_t size = sizeof(element_t);
    *element = (element_t *) malloc(size);
    if (element == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate element of size %zu bytes.", size),
                     NULL);
    }
    
    // Initialize
    (*element)->data = data;
    (*element)->next = NULL;
    
    return NULL;
}

void element_destroy(element_t *element)
{
    free(element);
}

error_t *queue_create(queue_t **queue)
{
    CHECK_NULL_ARGUMENT(queue, "queue");

    size_t size = sizeof(queue_t);
    *queue = (queue_t *) malloc(size);
    if (queue == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate queue of size %zu bytes.", size),
                     NULL);
    }
    
    //Initialize
    (*queue)->head = NULL;
    (*queue)->tail = NULL;
    (*queue)->size = 0;

    return NULL;
}

void queue_destroy(queue_t *queue)
{
    if (queue != NULL)
    {
        element_t *element = queue->head;
        while (element != NULL)
        {
            element_t *next = element->next;
            element_destroy(element);
            element = next;
        }
    }
    free(queue);
}

error_t *queue_enqueue(queue_t *queue, void *data)
{
    CHECK_NULL_ARGUMENT(queue, "queue");

    element_t *element;
    error_t *error = element_create(&element, data); 
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create element."),
                     error);
    }

    if (queue->head == NULL)
    {
        queue->head = element;
        queue->tail = element;
    }
    else
    {
        queue->tail->next = element;
        queue->tail = queue->tail->next;
    }
    queue->size++;

    return NULL;
}

error_t *queue_dequeue(queue_t *queue, void **data)
{
    CHECK_NULL_ARGUMENT(queue, "queue");
    CHECK_NULL_ARGUMENT(data, "data");

    if (queue->head == NULL)
    {
        return ERROR(ERROR_DESTROY,
                     string_create("failed to dequeue element from empty queue."),
                     NULL);
    }

    *data = queue->head->data;
    element_t *element = queue->head->next;
    element_destroy(queue->head); 
    queue->head = element;
    queue->size--;

    return NULL;
}