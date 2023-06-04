#include <device.h>
#include <cuda_device.h>
#include <cpu_device.h>

error_t *memory_allocate(void **p, size_t size, device_t device)
{
    if (p == NULL)
    {
        message_t message = create_message("received null pointer argument for 'p'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error;

    switch (device)
    {
    case DEVICE_CUDA:
        error = cuda_malloc(p, size);
        break;
    case DEVICE_CPU:
        error = cpu_malloc(p, size);
        break;
    default:
        message_t message = create_message("unknown device argument.");
        error = create_error(ERROR_UNKNOWN_DEVICE, __FILE__, __LINE__, __FUNCTION__, message, NULL);
        break;
    }

    if (error != NULL)
    {
        message_t message = create_message("failed to allocate %zu bytes on %s device.", size, device_string(device));
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, error);
    }
    
    return NULL;
}

error_t *memory_free(void *p, device_t device)
{
    if (p == NULL)
    {
        message_t message = create_message("received null pointer argument for 'p'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }
    
    error_t *error;

    switch (device)
    {
    case DEVICE_CUDA:
        error = cuda_free(p);
        break;
    case DEVICE_CPU:
        error = cpu_free(p);
        break;
    default:
        message_t message = create_message("unknown device argument.");
        error = create_error(ERROR_UNKNOWN_DEVICE, __FILE__, __LINE__, __FUNCTION__, message, NULL);
        break;
    }
    
    if (error != NULL)
    {
        message_t message = create_message("failed to free memory on %s device.", device_string(device));
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}

char *device_string(device_t device)
{
    switch (device)
    {
    case DEVICE_CPU:
        return "device_cpu"; 
    case DEVICE_CUDA:
        return "device_cuda";
    default:
        return "device_unknown";
    }
}