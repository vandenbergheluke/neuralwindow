#include <c_runtime.h>
#include <string.h>

error_t *c_malloc(void **pp, size_t size)
{
    CHECK_NULL(pp, "pp");

    *pp = malloc(size);
    if (*pp == NULL)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to allocate %zu bytes.", size), NULL);

    return NULL;
}

error_t *c_free(void *p)
{
    free(p);
    return NULL;
}

error_t *c_copy(const void *in_p, void *out_p, size_t size)
{
    CHECK_NULL(in_p, "in_p");
    CHECK_NULL(out_p, "out_p");

    memcpy(out_p, in_p, size);

    return NULL;
}

error_t *c_addition(datatype_t datatype, uint32_t size, const void *in_data_x, const void *in_data_y, void *out_data)
{
    CHECK_NULL(in_data_x, "in_data_x");
    CHECK_NULL(in_data_y, "in_data_y");
    CHECK_NULL(out_data, "out_data");

    switch (datatype)
    {
    case FLOAT32:
        float32_t *in_x = (float32_t *) in_data_x;
        float32_t *in_y = (float32_t *) in_data_y;
        float32_t *out = (float32_t *) out_data;
        for (uint32_t i = 0; i < size; i++)
            out[i] = in_x[i] + in_y[i];
        break;
    case FLOAT64:
        float64_t *in_x = (float64_t *) in_data_x;
        float64_t *in_y = (float64_t *) in_data_y;
        float64_t *out = (float64_t *) out_data;
        for (uint32_t i = 0; i < size; i++)
            out[i] = in_x[i] + in_y[i];
    default:
        return ERROR(ERROR_DATATYPE, create_string("Unsupported datatype %s", datatype_string(datatype)), NULL);    
    }

    return NULL;
}