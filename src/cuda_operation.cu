#include <cuda_runtime.h>
extern "C" {
    #include <cuda_operation.h>
}

extern "C" 
error_t cuda_malloc(buffer_t *buffer_ptr, size_t size)
{
    if (buffer_ptr == NULL)
    {
        return STATUS_NULL_POINTER;
    }
    cudaError_t error = cudaMalloc((void **) buffer_ptr, size);
    if (error != cudaSuccess)
    {
        printf("error:%s:%s:%d:%d:%s\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error));
        return STATUS_MEMORY_ALLOCATION_FAILURE;
    }
    return STATUS_SUCCESS;
}

extern "C"
error_t cuda_free(buffer_t buffer)
{
    cudaError_t error = cudaFree((void *) buffer);
    if (error != cudaSuccess)
    {
        printf("error:%s:%s:%d:%d:%s\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error));
        return STATUS_MEMORY_FREE_FAILURE;
    }
    return STATUS_SUCCESS;
}