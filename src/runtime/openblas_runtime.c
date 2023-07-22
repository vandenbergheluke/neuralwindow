#include <openblas_runtime.h>
#include <cblas.h>

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
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %s", datatype_string(datatype)),
                     NULL);    
    }

    return NULL;
}