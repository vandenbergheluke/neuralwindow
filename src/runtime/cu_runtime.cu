#include <cuda_runtime.h>
#include <cublas.h>
extern "C" {
    #include <cu_runtime.h>
}

extern "C" error_t *cu_malloc(void **pp, size_t size)
{
    CHECK_NULL(pp, "pp");

    cudaError_t error = cudaMallocManaged(pp, size);
    if (error != cudaSuccess)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to allocate %zu bytes, %s.", size, cudaGetErrorString(error)), NULL);

    return NULL;
}

extern "C" error_t *cu_free(void *p)
{
    CHECK_NULL(p, "p");

    cudaError_t error = cudaFree(p);
    if (error != cudaSuccess)
        return ERROR(ERROR_MEMORY_FREE, create_string("failed to free memory, %s.", cudaGetErrorString(error)), NULL);

    return NULL;
}

extern "C" error_t *cu_copy(const void *in_p, void *out_p, size_t size)
{
    CHECK_NULL(in_p, "in_p");
    CHECK_NULL(out_p, "out_p");

    cudaMemcpy(out_p, in_p, size, cudaMemcpyDeviceToDevice);

    return NULL;
}

extern "C" error_t *cu_addition(datatype_t datatype, uint32_t size, const void *in_data_x, const void *in_data_y, void *out_data)
{
    CHECK_NULL(in_data_x, "in_data_x");
    CHECK_NULL(in_data_y, "in_data_y");
    CHECK_NULL(out_data, "out_data");

    switch (datatype)
    {
    case FLOAT32:
        cublasScopy(size, (float32_t *) in_data_y, 1, (float32_t *) out_data, 1); 
        cublasSaxpy(size, 1.0, (float32_t *) in_data_x, 1, (float32_t *) out_data, 1);
        break;
    case FLOAT64:
        cublasDcopy(size, (float64_t *) in_data_y, 1, (float64_t *) out_data, 1);
        cublasDaxpy(size, 1.0, (float64_t *) in_data_x, 1, (float64_t *) out_data, 1);
    default:
        return ERROR(ERROR_DATATYPE, create_string("Unsupported datatype %s", datatype_string(datatype)), NULL);    
    }

    return NULL;
}
