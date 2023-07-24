#include <cuda_runtime.h>
#include <cublas.h>
extern "C" {
    #include <cu_runtime.h>
}

static cublasHandle_t handle = NULL;

extern "C" error_t *cu_create_context(void)
{
    cublasStatus_t status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create cuda context."),
                     NULL);
    }

    return NULL;
}

extern "C" void cu_destroy_context(void)
{
    // TODO: This can return an error but handling it makes tear downs awkward.
    // Most of the sample codes tend to ignore the error as well.
    // We should atleast print the error.
    cublasDestroy_v2(handle);
}

extern "C" error_t *cu_memory_allocate(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    cudaError_t error = cudaMallocManaged(pp, size);
    if (error != cudaSuccess)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes %s.", size, cudaGetErrorString(error)),
                     NULL);
    }

    return NULL;
}

extern "C" void cu_memory_free(void *p)
{
    // TODO: This can return an error but handling it makes tear downs awkward.
    // Most of the sample codes tend to ignore the error as well.
    // We should atleast print the error.
    cudaFree(p);
}

extern "C" error_t *cu_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    // TODO: The copy is annoying. Is there a way we can avoid this?
    float32_t alpha_32 = 1.0;
    float64_t alpha_64 = 1.0;
    switch (datatype)
    {
    case FLOAT32:
        cublasScopy_v2(handle, size, (float32_t *) y_data, 1, (float32_t *) z_data, 1); 
        cudaDeviceSynchronize();
        cublasSaxpy_v2(handle, size, &alpha_32, (float32_t *) x_data, 1, (float32_t *) z_data, 1);
        cudaDeviceSynchronize();
        break;
    case FLOAT64:
        cublasDcopy_v2(handle, size, (float64_t *) y_data, 1, (float64_t *) z_data, 1);
        cudaDeviceSynchronize();
        cublasDaxpy_v2(handle, size, &alpha_64, (float64_t *) x_data, 1, (float64_t *) z_data, 1);
        cudaDeviceSynchronize();
        break;
    default:
        return ERROR(ERROR_DATATYPE, 
                     string_create("unsupported datatype %s.", datatype_string(datatype)),
                     NULL);    
    }

    return NULL;
}

extern "C" error_t *cu_matrix_multiplication(datatype_t datatype,
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

    // Note: cuBLAS only accepts column major format hence, to get the
    // matrix product in row major we multiply the matrices that are in
    // row major in reverse order. (AB)^T = C^T = B^TA^T but B and A are 
    // already transposed if they are in row major.
    float32_t beta_32 = 0.0;
    float32_t alpha_32 = 1.0;
    float64_t beta_64 = 0.0;
    float64_t alpha_64 = 1.0;
    switch (datatype)
    {
    case FLOAT32:
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                       n, m, k, &alpha_32, (float32_t *) y_data, 
                       n, (float32_t *) x_data, k, &beta_32, (float32_t *) z_data, n);
        cudaDeviceSynchronize();
        break;
    case FLOAT64:
        cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                       n, m, k, &alpha_64, (float64_t *) y_data, 
                       n, (float64_t *) x_data, k, &beta_64, (float64_t *) z_data, n);
        cudaDeviceSynchronize();
        break;
    default:
        return ERROR(ERROR_DATATYPE, 
                     string_create("unsupported datatype %s.", datatype_string(datatype)),
                     NULL);    
    }

    return NULL;
}