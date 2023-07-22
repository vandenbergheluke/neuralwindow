#include <mkl_runtime.h>
#include <mkl.h>

error_t *mkl_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    switch (datatype)
    {
    case FLOAT32:
        // mkl_somatadd('R', 'N', 'N', size, 1, 1.0, (float32_t *) in_data_x, 1, 1.0, (float32_t *) in_data_y, 1, (float32_t *) out_data, 1);
        cblas_scopy(size, (float32_t *) y_data, 1, (float32_t *) z_data, 1); 
        cblas_saxpy(size, 1.0, (float32_t *) x_data, 1, (float32_t *) z_data, 1);
        break;
    case FLOAT64:
        // mkl_domatadd('R', 'N', 'N', size, 1, 1.0, (float64_t *) in_data_x, 1, 1.0, (float64_t *) in_data_y, 1, (float64_t *) out_data, 1);
        cblas_dcopy(size, (float64_t *) y_data, 1, (float64_t *) z_data, 1);
        cblas_daxpy(size, 1.0, (float64_t *) x_data, 1, (float64_t *) z_data, 1);
    default:
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d.", datatype),
                     NULL);    
    }

    return NULL;
}
