#include <id_pool.h>

static nw_error_t *id_node_create(id_node_t **id_node, uint64_t id, id_node_t *next)
{
    CHECK_NULL_ARGUMENT(id_node, "id_node");

    *id_node = (id_node_t *) malloc(sizeof(id_node_t));
    if (!*id_node)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(id_node_t)), NULL);
    }

    (*id_node)->id = id;
    (*id_node)->next = next;

    return NULL;
}

static void id_node_destroy(id_node_t *id_node)
{
    free(id_node);
}

nw_error_t *id_pool_create(id_pool_t **id_pool)
{
    CHECK_NULL_ARGUMENT(id_pool, "id_pool");

    *id_pool = (id_pool_t *) malloc(sizeof(id_pool_t));
    if (!*id_pool)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(id_node_t)), NULL);
    }

    (*id_pool)->head = NULL;
    (*id_pool)->size = 0;

    return NULL;
}

void id_pool_destroy(id_pool_t *id_pool)
{
    if (id_pool)
    {
        id_node_t *id_node = id_pool->head;
        while (id_node)
        {
            id_node_t *temp = id_node;
            id_node = id_node->next;
            id_node_destroy(temp);
        }
        free(id_pool);
    }
}

nw_error_t *id_pool_put(id_pool_t *id_pool, uint64_t id)
{
    nw_error_t *error = NULL;
    id_node_t *id_node = NULL;

    error = id_node_create(&id_node, id, id_pool->head);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create id node."), error);
    }
    
    id_pool->head = id_node;
    id_pool->size++;

    return NULL;
}

nw_error_t *id_pool_get(id_pool_t *id_pool, uint64_t *id)
{
    CHECK_NULL_ARGUMENT(id_pool, "id_pool");
    CHECK_NULL_ARGUMENT(id, "id");

    if (!id_pool->head)
    {
        return ERROR(ERROR_NULL, string_create("id pool is empty."), NULL);
    }

    *id = id_pool->head->id;
    id_node_t *id_node = id_pool->head;
    id_pool->head = id_pool->head->next;
    id_node_destroy(id_node);
    id_pool->size--;

    return NULL;
}

bool_t id_pool_is_empty(id_pool_t *id_pool)
{
    return !id_pool || !id_pool->head;
}

