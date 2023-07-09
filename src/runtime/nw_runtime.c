#include <nw_runtime.h>
#include <c_runtime.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime)
{
    CHECK_NULL_POINTER(pp, "pp");

    error_t *error;

    switch (runtime)
    {
    case C:
    case OPENBLAS:
    case MKL:
        error = c_malloc(pp, size);
        break;
    case CU:
        error = cu_malloc(pp, size);
        break;
    default:
        string_t message = create_string("unknown runtime argument.");
        error = create_error(ERROR_UNKNOWN_RUNTIME, __FILE__, __LINE__, __FUNCTION__, message, NULL);
        break;
    }

    if (error != NULL)
    {
        string_t message = create_string("failed to allocate %zu bytes for runtime %s.", size, runtime_string(runtime));
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, error);
    }
    
    return NULL;
}

error_t *nw_free(void *p, runtime_t runtime)
{
    error_t *error;

    switch (runtime)
    {
    case C:
    case OPENBLAS:
    case MKL:
        error = c_free(p);
        break;
    case CU:
        error = cu_free(p);
        break;
    default:
        string_t message = create_string("unknown runtime argument.");
        error = create_error(ERROR_UNKNOWN_RUNTIME, __FILE__, __LINE__, __FUNCTION__, message, NULL);
        break;
    }
    
    if (error != NULL)
    {
        string_t message = create_string("failed to free memory for runtime %s.", runtime_string(runtime));
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}

string_t runtime_string(runtime_t runtime)
{
    switch (runtime)
    {
    case C:
        return "C";
    case OPENBLAS:
        return "OPENBLAS"; 
    case MKL:
        return "MKL";
    case CU:
        return "CU";
    default:
        return NULL;
    }
}