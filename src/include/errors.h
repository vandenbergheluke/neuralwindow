#ifndef ERROR_H
#define ERROR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum error_t
{
    STATUS_SUCCESS,
    STATUS_MEMORY_ALLOCATION_FAILURE,
    STATUS_MEMORY_FREE_FAILURE,
    STATUS_UNKNOWN_DEVICE,
    STATUS_NULL_POINTER
} error_t;

char *get_error_string(error_t error);

#endif