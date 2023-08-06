#include <openblas_runtime.h>
#include <cblas.h>

error_t *openblas_memory_allocate(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    *pp = malloc(size);
    if (*pp == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    return NULL;
}

void openblas_memory_free(void *p)
{
    free(p);
}

error_t *openblas_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    switch (datatype)
    {
    case FLOAT32:
        cblas_scopy(size, (float32_t *) y_data, 1, (float32_t *) z_data, 1); 
        cblas_saxpy(size, 1.0, (float32_t *) x_data, 1, (float32_t *) z_data, 1);
        break;
    case FLOAT64:
        cblas_dcopy(size, (float64_t *) y_data, 1, (float64_t *) z_data, 1);
        cblas_daxpy(size, 1.0, (float64_t *) x_data, 1, (float64_t *) z_data, 1);
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %s", datatype_string(datatype)), NULL);    
    }

    return NULL;
}

error_t *openblas_matrix_multiplication(datatype_t datatype, uint32_t m, uint32_t k, uint32_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, const void *y_data, void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    switch (datatype)
    {
    case FLOAT32:
        cblas_sgemm(CblasRowMajor, (x_transpose) ? CblasNoTrans: CblasTrans, (y_transpose) ? CblasNoTrans : CblasTrans,
                    m, n, k, 1.0, (float32_t *) x_data, m, (float32_t *) y_data, k, 0.0, (float32_t *) z_data, m);
        break;
    case FLOAT64:
        cblas_dgemm(CblasRowMajor, (x_transpose) ? CblasNoTrans: CblasTrans, (y_transpose) ? CblasNoTrans : CblasTrans,
                    m, n, k, 1.0, (float64_t *) x_data, m, (float64_t *) y_data, k, 0.0, (float64_t *) z_data, m);
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);    
    }

    return NULL;
}

error_t *openblas_summation(datatype_t datatype, uint32_t axis, uint32_t current_dimension, uint32_t x_index, uint32_t *y_index, 
                            uint32_t *x_shape, uint32_t x_rank, uint32_t *x_strides, const void *x_data, void *y_data)
{
    CHECK_NULL_ARGUMENT(x_shape, "x_shape");
    CHECK_NULL_ARGUMENT(x_strides, "x_strides");
    CHECK_NULL_ARGUMENT(y_index, "y_index");
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");

    if (current_dimension >= x_rank)
    {
        return NULL;
    }

    error_t *error;

    if (current_dimension == axis)
    {
        error = cu_summation(datatype, axis, current_dimension + 1, x_index, y_index, x_shape, x_rank, x_strides, x_data, y_data);
        if (error != NULL)
        {
            return ERROR(ERROR_SUMMATION, string_create("failed to perform summation."), error);
        }
    }
    else
    {
        float32_t y_32 = 1.0;
        float64_t y_64 = 1.0;
        for (uint32_t i = 0; i < x_shape[current_dimension]; i++)
        {
            uint32_t j = x_index + i * x_strides[current_dimension];
            error = cu_summation(datatype, axis, current_dimension + 1, j, y_index, x_shape, x_rank, x_strides, x_data, y_data);
            if (error != NULL)
            {
                return ERROR(ERROR_SUMMATION, string_create("failed to perform summation."), error);
            }
            if (current_dimension == x_rank - 1 || current_dimension == x_rank - 2 && axis == x_rank - 1)
            {
                switch (datatype)
                {
                case FLOAT32:
                    ((float32_t *) y_data)[*y_index] = cblas_sdot(x_shape[axis], &((float32_t *) x_data)[j], x_strides[axis], &y_32, 0);
                    break;
                case FLOAT64:
                    ((float64_t *) y_data)[*y_index] = cblas_ddot(x_shape[axis], &((float64_t *) x_data)[j], x_strides[axis], &y_64, 0);
                    break;
                default:
                    return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %s.", datatype_string(datatype)), NULL);    
                }
                (*y_index)++;
            }
        }
    }

    return NULL;
}