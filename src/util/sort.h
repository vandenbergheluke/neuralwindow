#ifndef SORT_H
#define SORT_H

#include <errors.h>

nw_error_t *argument_sort(const int64_t *array, int64_t length, int64_t *sorted_array);
nw_error_t *descending_sort(const int64_t *array, int64_t length, int64_t *sorted_array);

#endif