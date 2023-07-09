#include <cuda_runtime.h>
extern "C" {
    #include <cu_runtime.h>
}

extern "C" error_t *cu_malloc(void **pp, size_t size)
{
    CHECK_NULL_POINTER(pp, "pp");

    cudaError_t error = cudaMalloc(pp, size);

    if (error != cudaSuccess)
    {
        string_t message = create_string("failed to allocate %zu bytes, %s.", size, cudaGetErrorString(error));
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    return NULL;
}

extern "C" error_t *cu_free(void *p)
{
    CHECK_NULL_POINTER(p, "p");

    cudaError_t error = cudaFree(p);

    if (error != cudaSuccess)
    {
        string_t message = create_string("failed to free memory, %s.", cudaGetErrorString(error));
        return create_error(ERROR_MEMORY_FREE, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    return NULL;
}