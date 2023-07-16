#include <queue.h>

error_t *create_queue(queue_t **queue)
{
    CHECK_NULL(queue, "queue");

    error_t *error;
    error = nw_malloc((void **) queue, sizeof(queue_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create queue."), error);
    
    //Initialize
    (*queue)->head = NULL;
    (*queue)->tail = NULL;
    
    return NULL;
}

error_t *destroy_queue(queue_t *queue)
{
    if (queue == NULL)
        return NULL;

    error_t *error;
    element_t *element;
    element_t *next;

    element = queue->head;
    while (element != NULL)
    {
        next = element->next;
        destroy_element(element);
        element = next;
    }

    error = nw_free((void *) queue, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy queue."), error);

    return NULL;
}

error_t *create_element(element_t **element, void *data)
{
    CHECK_NULL(element, "element");

    error_t *error;
    error = nw_malloc((void **) element, sizeof(queue_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create element."), error);
    
    // Initialize
    (*element)->data = data;
    (*element)->next = NULL;
    
    return NULL;
}

error_t *destroy_element(element_t *element)
{
    if (element == NULL)
        return NULL;

    error_t *error;
    error = nw_free((void *) element, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy element."), error);

    return NULL;
}

error_t *enqueue(queue_t *queue, void *data)
{
    CHECK_NULL(queue, "queue");

    error_t *error;
    element_t *element;
    error = create_element(&element, data); 
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to enqueue element."), error);

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

    return NULL;
}

error_t *dequeue(queue_t *queue, void **data)
{
    CHECK_NULL(queue, "queue");
    CHECK_NULL(data, "data");

    error_t *error;
    element_t *element;

    if (queue->head == NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to dequeue element from empty queue."), NULL);

    *data = queue->head->data;
    element = queue->head->next;
    error = destroy_element(queue->head); 
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to dequeue element."), error);

    queue->head = element;

    return NULL;
}