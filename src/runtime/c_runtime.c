#include <c_runtime.h>
#include <string.h>

error_t *c_malloc(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    *pp = malloc(size);
    if (*pp == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes.", size),
                     NULL);
    }

    return NULL;
}

void c_free(void *p)
{
    free(p);
}

error_t *c_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    switch (datatype)
    {
    case FLOAT32:;
        float32_t *x_data_float32 = (float32_t *) x_data;
        float32_t *y_data_float32 = (float32_t *) y_data;
        float32_t *z_data_float32 = (float32_t *) z_data;
        for (uint32_t i = 0; i < size; i++)
        {
            z_data_float32[i] = x_data_float32[i] + y_data_float32[i];
        }
        break;
    case FLOAT64:;
        float64_t *x_data_float64 = (float64_t *) x_data;
        float64_t *y_data_float64 = (float64_t *) y_data;
        float64_t *z_data_float64 = (float64_t *) z_data;
        for (uint32_t i = 0; i < size; i++)
        {
            z_data_float64[i] = x_data_float64[i] + y_data_float64[i];
        }
        break;
    default:
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d.", datatype),
                     NULL);    
    }

    return NULL;
}