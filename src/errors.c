#include <errors.h>

char *get_error_string(error_t error)
{
    switch (error)
    {
        case STATUS_SUCCESS:
            return "success";
        case STATUS_MEMORY_ALLOCATION_FAILURE:
            return "failed to allocate sufficient memory";
        case STATUS_MEMORY_FREE_FAILURE:
            return "failed to free memory";
        case STATUS_NULL_POINTER:
            return "received null pointer";
        default:
            return "unknown status";
    }
}
