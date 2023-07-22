#include <init.h>

error_t *init_zeroes(tensor_t *x)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->data, "x->buffer->data");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");

    size_t size = view_size(x->buffer->view) * datatype_size(x->buffer->datatype);
    switch (x->buffer->datatype)
    {
    case FLOAT32:
        memset(x->buffer->data, (float32_t) 0.0, size); 
        break;
    case FLOAT64:
        memset(x->buffer->data, (float64_t) 0.0, size); 
        break;
    default:
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d.", x->buffer->datatype),
                     NULL);
    }

    return NULL;
}

error_t *init_ones(tensor_t *x)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->data, "x->buffer->data");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");

    size_t size = view_size(x->buffer->view) * datatype_size(x->buffer->datatype);
    switch (x->buffer->datatype)
    {
    case FLOAT32:
        memset(x->buffer->data, (float32_t) 1.0, size); 
        break;
    case FLOAT64:
        memset(x->buffer->data, (float64_t) 1.0, size); 
        break;
    default:
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d.", x->buffer->datatype),
                     NULL);
    }

    return NULL;
}
