#ifndef ERROR_H
#define ERROR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef char * message_t;

typedef enum error_type_t
{
    ERROR_MEMORY_ALLOCATION,
    ERROR_MEMORY_FREE,
    ERROR_UNKNOWN_DEVICE,
    ERROR_NULL_POINTER,
    ERROR_DATATYPE_CONFLICT,
    ERROR_SHAPE_CONFLICT,
    ERROR_CREATE,
    ERROR_DESTROY
} error_type_t;

typedef struct error_t
{
    error_type_t type;
    const char *file;
    unsigned int line_number;
    const char *function;
    message_t message;
    struct error_t *next_error;
} error_t;

message_t create_message(const char *format, ...);
void destory_message(message_t message);
error_t *create_error(error_type_t type, const char *file, unsigned int line_number, const char *function, message_t message, error_t *next_error);
void destroy_error(error_t *error);
void print_error(error_t *error);
char *error_type_string(error_type_t error_type);

#endif