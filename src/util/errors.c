#include <errors.h>

error_t *error_create(error_type_t error_type, string_t file, uint32_t line_number, string_t function, string_t message, error_t *next_error)
{
    error_t *error = (error_t *) malloc(sizeof(error_t));
    if (error == NULL)
        return NULL;

    error->error_type = error_type;
    error->file = file;
    error->line_number = line_number;
    error->function = function;
    error->message = message;
    error->next_error = next_error;
    return error;
}

void error_destroy(error_t *error)
{
    while(error != NULL)
    {
        error_t *next_error = error->next_error;
        string_destroy(error->message);
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
    case ERROR_UKNOWN_OPERATION_TYPE:
        return "ERROR_UKNOWN_OPERATION_TYPE";
    case ERROR_NULL:
        return "ERROR_NULL";
    case ERROR_DATATYPE_CONFLICT:
        return "ERROR_DATATYPE_CONFLICT";
    case ERROR_SHAPE_CONFLICT:
        return "ERROR_SHAPE_CONFLICT";
    case ERROR_RUNTIME_CONFLICT:
        return "ERROR_RUNTIME_CONFLICT";
    case ERROR_CREATE:
        return "ERROR_CREATE";
    case ERROR_DESTROY:
        return "ERROR_DESTROY";
    case ERROR_BROADCAST:
        return "ERROR_BROADCAST";
    case ERROR_INITIALIZATION:
        return "ERROR_INITIALIZATION";
    case ERROR_DATATYPE:
        return "ERROR_DATATYPE";
    case ERROR_COPY:
        return "ERROR_COPY";
    case ERROR_ADDITION:
        return "ERROR_ADDITION";
    case ERROR_CONTIGUOUS:
        return "ERROR_CONTIGUOUS";
    case ERROR_FORWARD:
        return "ERROR_FORWARD";
    case ERROR_BACKWARD:
        return "ERROR_BACKWARD";
    case ERROR_SET:
        return "ERROR_SET";
    case ERROR_OVERFLOW:
        return "ERROR_OVERFLOW";
    case ERROR_EXPAND:
        return "ERROR_EXPAND";
    default:
        return "ERROR";
    }
}

void error_print(error_t *error)
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