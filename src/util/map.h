#ifndef MAP_H
#define MAP_H

#include <datatype.h>
#include <errors.h>

typedef struct entry_t
{
    string_t key;
    void *data;
} entry_t;

typedef struct map_t
{
    entry_t *entries;
    uint64_t capacity;
    uint64_t length;
} map_t;

#define INITIAL_CAPACITY 16

error_t *map_create(map_t **map);
void map_destroy(map_t *map);
error_t *map_get(map_t *map, string_t key, void **data);
bool_t map_contains(map_t *map, string_t key);
error_t *map_set(map_t *map, string_t key, void *data);

#endif