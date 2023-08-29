#include <map.h>

// djb2
static uint64_t map_hash_key(string_t string)
{
    uint64_t hash = 5381;
    int c;

    while ((c = *string++))
    {
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

error_t *map_create(map_t **map)
{
    CHECK_NULL_ARGUMENT(map, "map");

    size_t size = sizeof(map_t);
    *map = (map_t *) malloc(size);
    if (*map == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate map of size %zu bytes.", size),
                     NULL);
    }
    (*map)->length = 0;
    (*map)->capacity = INITIAL_CAPACITY;
    size = (*map)->capacity * sizeof(entry_t);
    (*map)->entries = (entry_t *) malloc(size);
    for (uint64_t i = 0; i < (*map)->capacity; ++i)
    {
        (*map)->entries[i].key = NULL;
        (*map)->entries[i].data = NULL;
    }

    if ((*map)->entries == NULL)
    {
        free(*map);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate map->entries of size %zu bytes.", size),
                     NULL);
    }

    return NULL;
}

void map_destroy(map_t *map)
{
    if (map != NULL)
    {
        if (map->entries != NULL)
        {
            for (uint64_t i = 0; i < map->capacity; ++i)
            {
                string_destroy(map->entries[i].key);
            }
        }
        free(map->entries);
    }
    free(map);
}

error_t *map_get(map_t *map, string_t key, void **data)
{
    CHECK_NULL_ARGUMENT(map, "map");
    CHECK_NULL_ARGUMENT(map->entries, "map->entries");
    CHECK_NULL_ARGUMENT(key, "key");
    CHECK_NULL_ARGUMENT(data, "data");

    uint64_t hash = map_hash_key(key);
    uint64_t index = (uint64_t)(hash & (uint64_t)(map->capacity - 1));
    while (map->entries[index].key != NULL)
    {
        if (strcmp(key, map->entries[index].key) == 0)
        {
            (*data) = map->entries[index].data;
        }
        ++index;
        if (index >= map->capacity)
        {
            index = 0;
        }
    }
    return NULL;
}

static error_t *map_set_entry(entry_t *entries, uint64_t capacity, string_t key, void *data)
{
    CHECK_NULL_ARGUMENT(entries, "entries");
    CHECK_NULL_ARGUMENT(key, "key");

    uint64_t hash = map_hash_key(key);
    uint64_t index = (uint64_t)(hash & (uint64_t)(capacity - 1));
    while (entries[index].key != NULL)
    {
        if (strcmp(key, entries[index].key) == 0)
        {
            entries[index].data = data;
        }
        ++index;
        if (index >= capacity)
        {
            index = 0;
        }
    }

    entries[index].key = key;
    entries[index].data = data;

    return NULL;
}

bool_t map_contains(map_t *map, string_t key)
{
    if (map == NULL || map->entries == NULL || key == NULL)
    {
        return false;
    }

    uint64_t hash = map_hash_key(key);
    uint64_t index = (uint64_t)(hash & (uint64_t)(map->capacity - 1));
    while (map->entries[index].key != NULL)
    {
        if (strcmp(key, map->entries[index].key) == 0)
        {
            return true;
        }
        ++index;
        if (index >= map->capacity)
        {
            index = 0;
        }
    }
    return false;
}

static error_t *map_expand(map_t *map)
{
    CHECK_NULL_ARGUMENT(map, "map");
    CHECK_NULL_ARGUMENT(map->entries, "map->entries");

    uint64_t new_capacity = map->capacity * 2;
    if (new_capacity < map->capacity)
    {
        return ERROR(ERROR_OVERFLOW,
                     string_create("capacity %lu is too large to be doubled.",
                     map->capacity), NULL);
    }

    size_t size = new_capacity * sizeof(entry_t);
    entry_t *new_entries = (entry_t *) malloc(size);
    if (new_entries == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate map entries of size %zu bytes.", size),
                     NULL);
    }
    for (uint64_t i = 0; i < new_capacity; ++i)
    {
        new_entries[i].key = NULL;
        new_entries[i].data = NULL;
    }
    
    for (uint64_t i = 0; i < map->capacity; ++i)
    {
        entry_t entry = map->entries[i];
        if (entry.key != NULL)
        {
            error_t *error = map_set_entry(new_entries, new_capacity, entry.key, entry.data);
            if (error != NULL)
            {
                free(new_entries);
                return ERROR(ERROR_SET,
                             string_create("failed to set entry with corresponding key %s.", entry.key),
                             NULL);
            }
        }
    }

    free(map->entries);
    map->entries = new_entries;
    map->capacity = new_capacity;

    return NULL;
}

error_t *map_set(map_t *map, string_t key, void *data)
{
    CHECK_NULL_ARGUMENT(map, "map");
    CHECK_NULL_ARGUMENT(map->entries, "map->entries");
    CHECK_NULL_ARGUMENT(key, "key");

    error_t *error;
    if (map->length >= map->capacity / 2)
    {
        error = map_expand(map);
        if (error != NULL)
        {
            return ERROR(ERROR_EXPAND,
                         string_create("failed to increase capacity of map."),
                         error);
        }
    }

    error = map_set_entry(map->entries, map->capacity, key, data);
    if (error != NULL)
    {
        return ERROR(ERROR_SET,
                     string_create("failed to set map entry with corresponding key %s.", key),
                     error);
    }
    ++map->length;

    return NULL;
}
