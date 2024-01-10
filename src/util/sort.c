#include <sort.h>

typedef struct pair_t
{
    int64_t index;
    int64_t value;
} pair_t;
    
static int pair_compare(const void *a, const void *b)
{
    pair_t *pair_a = (pair_t *) a;
    pair_t *pair_b = (pair_t *) b;

    return (pair_a->value - pair_b->value);
}

nw_error_t *argument_sort(const int64_t *array, int64_t length, int64_t *sorted_array)
{
    CHECK_NULL_ARGUMENT(array, "axis");
    CHECK_NULL_ARGUMENT(sorted_array, "sorted_array");

    pair_t pairs[length];
    for (int64_t i = 0; i < length; ++i)
    {
        pairs[i].index = i;
        pairs[i].value = array[i];
    }

    qsort((void *) pairs, (size_t) length, sizeof(pair_t), pair_compare);

    for (int64_t i = 0; i < length; ++i)
    {
        sorted_array[i] = pairs[i].index;
    }

    return NULL;
}

static int compare(const void * p1, const void * p2)
{
    return (*(int64_t *) p2 - *(int64_t *) p1);
}

nw_error_t *descending_sort(const int64_t *array, int64_t length, int64_t *sorted_array)
{
    CHECK_NULL_ARGUMENT(array, "axis");
    CHECK_NULL_ARGUMENT(sorted_array, "sorted_array");

    for (int64_t i = 0; i < length; ++i)
    {
        sorted_array[i] = array[i];
    }

    qsort(sorted_array, length, sizeof(int64_t), compare);

    return NULL;
}