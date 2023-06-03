#include <device.h>
#include <cuda_operation.h>
#include <cpu_operation.h>

error_t memory_allocate(buffer_t *buffer_ptr, size_t size, device_t device)
{
    switch (device)
    {
    case DEVICE_CUDA:
        return cuda_malloc(buffer_ptr, size);
    case DEVICE_CPU:
        return cpu_malloc(buffer_ptr, size);
    default:
        return STATUS_UNKNOWN_DEVICE;
    }
}

error_t memory_free(buffer_t buffer, device_t device)
{
    switch (device)
    {
    case DEVICE_CUDA:
        return cuda_free(buffer);
    case DEVICE_CPU:
        return cpu_free(buffer);
    default:
        return STATUS_UNKNOWN_DEVICE;
    }
}