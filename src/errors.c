#include <errors.h>
#include <stdarg.h>

message_t create_message(const char *format, ...)
{
    if (format == NULL)
    {
        return NULL;
    }

    va_list arguments;

    va_start(arguments, format);
    size_t size = vsnprintf(NULL, 0, format, arguments) + 1;
    va_end(arguments);

    char *message = (char *) malloc(size);
    
    va_start(arguments, format);
    vsnprintf(message, size, format, arguments);
    va_end(arguments);

    return message;
}

void destory_message(message_t message)
{
    free(message);
}

error_t *create_error(error_type_t type, const char *file, unsigned int line_number, const char *function, message_t message, error_t *next_error)
{
    error_t *error = (error_t *) malloc(sizeof(error_t));
    if (error == NULL)
    {
        return NULL;
    }

    error->type = type;
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
        destory_message(error->message);
        free(error);
        error = next_error;
    }
}

char *error_type_string(error_type_t error_type)
{
    switch (error_type)
    {
    case ERROR_MEMORY_ALLOCATION:
        return "error_memory_allocation"; 
    case ERROR_MEMORY_FREE:
        return "error_memory_free";
    case ERROR_UNKNOWN_DEVICE:
        return "error_uknown_device";
    case ERROR_NULL_POINTER:
        return "error_null_pointer";
    case ERROR_DATATYPE_CONFLICT:
        return "error_datatype_conflict";
    case ERROR_SHAPE_CONFLICT:
        return "error_shape_conflict";
    case ERROR_CREATE:
        return "error_create";
    case ERROR_DESTROY:
        return "error_destroy";
    default:
        return "error_unknown";
    }
}

void print_error(error_t *error)
{
    while (error != NULL)
    {
        fprintf(
            stderr, 
            "%s:%s:%u:%s:%s\n", 
            error_type_string(error->type),
            error->file != NULL ? error->file : "file_unknown", 
            error->line_number, 
            error->function != NULL ? error->function : "function_unknown", 
            error->message != NULL ? error->message : "message_unknown"
        );
        error = error->next_error;
    }
}