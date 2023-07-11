#include <cuda_runtime.h>
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