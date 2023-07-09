#include <errors.h>

error_t *create_error(error_type_t error_type, string_t file, uint32_t line_number, string_t function, string_t message, error_t *next_error)
{
    error_t *error = (error_t *) malloc(sizeof(error_t));
    if (error == NULL)
    {
        return NULL;
    }

    error->error_type = error_type;
    error->file = file;
    error->line_number = line_number;
    error->function = function;
    error->message = message;
    error->next_error = next_error;
    return error;
}

void destroy_error(error_t *error)
{
    while(error != NULL)
    {
        error_t *next_error = error->next_error;
        destroy_string(error->message);
        free(error);
        error = next_error;
    }
}

string_t error_type_string(error_type_t error_type)
{
    switch (error_type)
    {
    case ERROR_MEMORY_ALLOCATION:
        return "ERROR_MEMORY_ALLOCATION"; 
    case ERROR_MEMORY_FREE:
        return "ERROR_MEMORY_FREE";
    case ERROR_UNKNOWN_RUNTIME:
        return "ERROR_UNKNOWN_RUNTIME";
    case ERROR_NULL_POINTER:
        return "ERROR_NULL_POINTER";
    case ERROR_DATATYPE_CONFLICT:
        return "ERROR_DATATYPE_CONFLICT";
    case ERROR_SHAPE_CONFLICT:
        return "ERROR_SHAPE_CONFLICT";
    case ERROR_CREATE:
        return "ERROR_CREATE";
    case ERROR_DESTROY:
        return "ERROR_DESTROY";
    case ERROR_BROADCAST:
        return "ERROR_BROADCAST";
    case ERROR_INITIALIZATION:
        return "ERROR_INITIALIZATION";
    case ERROR_UNKNOWN_INSTANCE_TYPE:
        return "ERROR_UNKNOWN_INSTANCE_TYPE";
    default:
        return "ERROR";
    }
}

void print_error(error_t *error)
{
    while (error != NULL)
    {
        fprintf(
            stderr, 
            "%s:%s:%u:%s:%s\n", 
            error_type_string(error->error_type),
            error->file != NULL ? error->file : "NULL", 
            error->line_number, 
            error->function != NULL ? error->function : "NULL", 
            error->message != NULL ? error->message : "NULL"
        );
        error = error->next_error;
    }
}