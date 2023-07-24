#include <mkl_runtime.h>
#include <mkl.h>

error_t *mkl_memory_allocate(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    *pp = mkl_malloc(size, 64);
    if (*pp == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes.", size),
                     NULL);
    }

    return NULL;
}

void mkl_memory_free(void *p)
{
    mkl_free(p);
}

error_t *mkl_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data)
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
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d.", (int) datatype),
                     NULL);    
    }

    return NULL;
}

error_t *mkl_matrix_multiplication(datatype_t datatype,
                                   uint32_t m,
                                   uint32_t k,
                                   uint32_t n, 
                                   const void *x_data,
                                   const void *y_data,
                                   void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    switch (datatype)
    {
    case FLOAT32:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, (float32_t *) x_data, m, 
                    (float32_t *) y_data, k, 0.0, (float32_t *) z_data, m);
        break;
    case FLOAT64:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, (float64_t *) x_data, m, 
                    (float64_t *) y_data, k, 0.0, (float64_t *) z_data, m);
        break;
    default:
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d.", (int) datatype),
                     NULL);    
    }

    return NULL;
}
