#include <mkl_runtime.h>
#include <mkl.h>

error_t *mkl_addition(datatype_t datatype, uint32_t size, const void *in_data_x, const void *in_data_y, void *out_data)
{
    CHECK_NULL(in_data_x, "in_data_x");
    CHECK_NULL(in_data_y, "in_data_y");
    CHECK_NULL(out_data, "out_data");

    switch (datatype)
    {
    case FLOAT32:
        // mkl_somatadd('R', 'N', 'N', size, 1, 1.0, (float32_t *) in_data_x, 1, 1.0, (float32_t *) in_data_y, 1, (float32_t *) out_data, 1);
        cblas_scopy(size, (float32_t *) in_data_y, 1, (float32_t *) out_data, 1); 
        cblas_saxpy(size, 1.0, (float32_t *) in_data_x, 1, (float32_t *) out_data, 1);
        break;
    case FLOAT64:
        // mkl_domatadd('R', 'N', 'N', size, 1, 1.0, (float64_t *) in_data_x, 1, 1.0, (float64_t *) in_data_y, 1, (float64_t *) out_data, 1);
        cblas_dcopy(size, (float64_t *) in_data_y, 1, (float64_t *) out_data, 1);
        cblas_daxpy(size, 1.0, (float64_t *) in_data_x, 1, (float64_t *) out_data, 1);
    default:
        return ERROR(ERROR_DATATYPE, create_string("Unsupported datatype %s", datatype_string(datatype)), NULL);    
    }

    return NULL;
}
