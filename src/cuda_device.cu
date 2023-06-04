#include <cuda_runtime.h>
extern "C" {
    #include <cuda_device.h>
}

extern "C" 
error_t *cuda_malloc(void **p, size_t size)
{
    if (p == NULL)
    {
        message_t message = create_message("received null pointer argument for 'p'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    cudaError_t error = cudaMalloc(p, size);

    if (error != cudaSuccess)
    {
        message_t message = create_message("failed to allocate %zu bytes, %s.", size, cudaGetErrorString(error));
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    return NULL;
}

extern "C"
error_t *cuda_free(void *p)
{
    if (p == NULL)
    {
        message_t message = create_message("received null pointer argument for 'p'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    cudaError_t error = cudaFree(p);

    if (error != cudaSuccess)
    {
        message_t message = create_message("failed to free 'p'.");
        return create_error(ERROR_MEMORY_FREE, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    return NULL;
}