#ifndef ERRORS_H
#define ERRORS_H

#include <stdio.h>
#include <string.h>
#include <datatype.h>

typedef enum error_type_t
{
    ERROR_MEMORY_ALLOCATION,
    ERROR_MEMORY_FREE,
    ERROR_UNKNOWN_RUNTIME,
    ERROR_UKNOWN_OPERATION_TYPE,
    ERROR_NULL,
    ERROR_DATATYPE_CONFLICT,
    ERROR_SHAPE_CONFLICT,
    ERROR_RUNTIME_CONFLICT,
    ERROR_RANK_CONFLICT,
    ERROR_CREATE,
    ERROR_DESTROY,
    ERROR_BROADCAST,
    ERROR_INITIALIZATION,
    ERROR_DATATYPE,
    ERROR_COPY,
    ERROR_ADDITION,
    ERROR_CONTIGUOUS,
    ERROR_FORWARD,
    ERROR_BACKWARD,
    ERROR_SET,
    ERROR_OVERFLOW,
    ERROR_EXPAND,
    ERROR_PERMUTE,
    ERROR_SUMMATION,
    ERROR_SQUARE_ROOT,
    ERROR_RESHAPE,
    ERROR_SLICE,
    ERROR_BINARY_ELEMENTWISE,
    ERROR_REDUCTION
} error_type_t;

typedef struct error_t
{
    error_type_t error_type;
    string_t file;
    uint32_t line_number;
    string_t function;
    string_t message;
    struct error_t *next_error;
} error_t;

error_t *error_create(error_type_t error_type, string_t file, uint32_t line_number, string_t function, string_t message, error_t *next_error);
void error_destroy(error_t *error);
void error_print(error_t *error);
string_t error_type_string(error_type_t error_type);

#define ERROR(error_type, error_string, error) (error_create(error_type, __FILE__, __LINE__, __FUNCTION__, error_string, error))

#define CHECK_NULL_ARGUMENT(pointer, string) ({\
            if (pointer == NULL)\
            {\
                return ERROR(ERROR_NULL, string_create("received null argument for %s.", string), NULL);\
            }\
        })

#endif