#ifndef ID_POOL_H
#define ID_POOL_H

#include <datatype.h>
#include <errors.h>

typedef struct id_node_t
{
    uint64_t id;
    struct id_node_t *next;
} id_node_t;

typedef struct id_pool_t
{
    uint64_t size;
    id_node_t *head;    
} id_pool_t;

nw_error_t *id_pool_create(id_pool_t **id_pool);
void id_pool_destroy(id_pool_t *id_pool);
nw_error_t *id_pool_put(id_pool_t *id_pool, uint64_t id);
nw_error_t *id_pool_get(id_pool_t *id_pool, uint64_t *id);
bool_t id_pool_is_empty(id_pool_t *id_pool);

#endif