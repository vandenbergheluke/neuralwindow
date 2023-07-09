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
    ERROR_NULL_POINTER,
    ERROR_DATATYPE_CONFLICT,
    ERROR_SHAPE_CONFLICT,
    ERROR_CREATE,
    ERROR_DESTROY,
    ERROR_BROADCAST,
    ERROR_INITIALIZATION,
    ERROR_UNKNOWN_INSTANCE_TYPE
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


error_t *create_error(error_type_t error_type, string_t file, uint32_t line_number, string_t function, string_t message, error_t *next_error);
void destroy_error(error_t *error);
void print_error(error_t *error);
string_t error_type_string(error_type_t error_type);

#define CHECK_NULL_POINTER(p, s) ({\
            if (p == NULL)\
            {\
                string_t message = create_string("received null pointer argument for %s.", s);\
                return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);\
            }\
        })

#endif