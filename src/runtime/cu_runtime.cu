#include <cuda_runtime.h>
#include <cublas.h>
extern "C" {
    #include <cu_runtime.h>
}

extern "C" error_t *cu_malloc(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    cudaError_t error = cudaMallocManaged(pp, size);
    if (error != cudaSuccess)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes, %s.", size, cudaGetErrorString(error)),
                     NULL);
    }

    return NULL;
}

extern "C" void cu_free(void *p)
{
    cudaFree(p);
}

extern "C" error_t *cu_copy(const void *src, void *dst, size_t size)
{
    CHECK_NULL_ARGUMENT(src, "src");
    CHECK_NULL_ARGUMENT(dst, "dst");

    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);

    return NULL;
}

extern "C" error_t *cu_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data)
{
    CHECK_NULL_ARGUMENT(x_data, "x_data");
    CHECK_NULL_ARGUMENT(y_data, "y_data");
    CHECK_NULL_ARGUMENT(z_data, "z_data");

    switch (datatype)
    {
    case FLOAT32:
        cublasScopy(size, (float32_t *) y_data, 1, (float32_t *) z_data, 1); 
        cublasSaxpy(size, 1.0, (float32_t *) x_data, 1, (float32_t *) z_data, 1);
        break;
    case FLOAT64:
        cublasDcopy(size, (float64_t *) y_data, 1, (float64_t *) z_data, 1);
        cublasDaxpy(size, 1.0, (float64_t *) x_data, 1, (float64_t *) z_data, 1);
        break;
    default:
        return ERROR(ERROR_DATATYPE, 
                     string_create("Unsupported datatype %s", datatype_string(datatype)),
                     NULL);    
    }

    return NULL;
}
