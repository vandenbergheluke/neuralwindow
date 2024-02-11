/**@file datatype.c
 * @brief Implements utilities that interact with common datatypes.
 *
 */

#include <datatype.h>
#include <math.h>

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
    if (!format)
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

bool_t is_zero(void *value, datatype_t datatype)
{
    if (!value)
    {
        return true;
    }

    switch (datatype)
    {
    case FLOAT32:
        return fabsf(*(float32_t *) value) < FLT_EPSILON;
        break;
    case FLOAT64:
        return fabs(*(float64_t *) value) < DBL_EPSILON;
        break;
    }

    return false;
}

bool_t compare_greater_than_equal(void *lvalue, void *rvalue, datatype_t datatype)
{
    switch (datatype)
    {
    case FLOAT32:
        return *(float32_t *) lvalue >= *(float32_t *) rvalue;
    case FLOAT64:
        return *(float64_t *) lvalue >= *(float64_t *) rvalue;
    }

    return false;
}