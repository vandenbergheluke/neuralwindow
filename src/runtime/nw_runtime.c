#include <nw_runtime.h>
#include <c_runtime.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime)
{
    CHECK_NULL(pp, "pp");

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
        error = ERROR(ERROR_UNKNOWN_RUNTIME, create_string("unknown runtime argument."), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to allocate %zu bytes for runtime %s.", size, runtime_string(runtime)), error);
    
    return NULL;
}

error_t *nw_free(void *p, runtime_t runtime)
{
    if (p == NULL)
        return NULL;

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
        error = ERROR(ERROR_UNKNOWN_RUNTIME, create_string("unknown runtime argument."), NULL);
        break;
    }
    
    if (error != NULL)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to free memory for runtime %s.", runtime_string(runtime)), error);

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