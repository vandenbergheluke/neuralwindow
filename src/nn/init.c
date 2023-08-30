#include <init.h>
#include <datatype.h>
#include <string.h>

nw_error_t *init_zeroes(tensor_t *x)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->data, "x->buffer->data");

    switch (x->buffer->datatype)
    {
    case FLOAT32:
        memset(x->buffer->data, (float32_t) 0.0, x->buffer->size); 
        break;
    case FLOAT64:
        memset(x->buffer->data, (float64_t) 0.0, x->buffer->size); 
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", x->buffer->datatype), NULL);
    }

    return NULL;
}

nw_error_t *init_ones(tensor_t *x)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->data, "x->buffer->data");

    switch (x->buffer->datatype)
    {
    case FLOAT32:
        memset(x->buffer->data, (float32_t) 1.0, x->buffer->size); 
        break;
    case FLOAT64:
        memset(x->buffer->data, (float64_t) 1.0, x->buffer->size); 
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", x->buffer->datatype), NULL);
    }

    return NULL;
}

