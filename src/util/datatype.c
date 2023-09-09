/**@file datatype.c
 * @brief Implements utilities that interact with common datatypes.
 *
 */

#include <datatype.h>

string_t datatype_string(datatype_t datatype)
{
    switch (datatype)
    {
    case FLOAT32:
        return "FLOAT32";
    case FLOAT64:
        return "FLOAT64";
    default:
        return "UNKNOWN";
    }
}

size_t datatype_size(datatype_t datatype)
{
    switch (datatype)
    {
    case FLOAT32:
        return sizeof(float32_t);
    case FLOAT64:
        return sizeof(float64_t);
    default:
        return 0;
    }
}

string_t string_create(string_t format, ...)
{
    if (format == NULL)
    {
        return NULL;
    }

    va_list arguments;

    va_start(arguments, format);
    size_t size = vsnprintf(NULL, 0, format, arguments) + 1;
    va_end(arguments);

    char *string = (char *) malloc(size);
    
    va_start(arguments, format);
    vsnprintf(string, size, format, arguments);
    va_end(arguments);

    return string;
}

void string_destroy(string_t string)
{
    free((char *) string);
}
