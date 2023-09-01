#include <cuda_runtime.h>
#include <cublas.h>
extern "C" {
    #include <cu_runtime.h>
}

static cublasHandle_t handle = NULL;

extern "C" nw_error_t *cu_create_context(void)
{
    cublasStatus_t status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create cuda context."), NULL);
    }

    return NULL;
}

extern "C" void cu_destroy_context(void)
{
    cublasDestroy_v2(handle);
}

extern "C" nw_error_t *cu_memory_allocate(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    cudaError_t error = cudaMallocManaged(pp, size);
    if (error != cudaSuccess)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes %s.", size, cudaGetErrorString(error)), NULL);
    }

    return NULL;
}

extern "C" void cu_memory_free(void *p)
{
    cudaFree(p);
}

extern "C" static void cu_exponential_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = expf(x_data[i * x_stride]); 
    }
}

extern "C" static void cu_exponential_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = exp(x_data[i * x_stride]); 
    }
}

extern "C" void cu_exponential(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_exponential_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_exponential_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_logarithm_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = logf(x_data[i * x_stride]); 
    }
}

extern "C" static void cu_logarithm_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = log(x_data[i * x_stride]); 
    }
}

extern "C" void cu_logarithm(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_logarithm_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_logarithm_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_sine_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sinf(x_data[i * x_stride]); 
    }
}

extern "C" static void cu_sine_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sin(x_data[i * x_stride]); 
    }
}

extern "C" void cu_sine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_sine_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_sine_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_cosine_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = cosf(x_data[i * x_stride]); 
    }
}

extern "C" static void cu_cosine_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = cos(x_data[i * x_stride]); 
    }
}

extern "C" void cu_cosine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_cosine_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_cosine_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_square_root_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sqrtf(x_data[i * x_stride]); 
    }
}

extern "C" static void cu_square_root_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sqrt(x_data[i * x_stride]); 
    }
}

extern "C" void cu_square_root(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_square_root_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_square_root_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_reciprocal_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = 1. / x_data[i * x_stride]; 
    }
}

extern "C" static void cu_reciprocal_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = 1. / x_data[i * x_stride]; 
    }
}

extern "C" void cu_reciprocal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_reciprocal_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_reciprocal_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" void cu_copy(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cublasScopy_v2(handle, (int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        cudaDeviceSynchronize();
        break;
    case FLOAT64:
        cublasDcopy_v2(handle, (int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }
}

extern "C" static void cu_negation_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    float32_t alpha = -1.0;
    cudaMemset(y_data, 0, n * sizeof(float32_t));
    cublasSaxpy_v2(handle, (int) n, &alpha, x_data, (int) x_stride, y_data, (int) y_stride);
    cudaDeviceSynchronize();
}

extern "C" static void cu_negation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    float64_t alpha = -1.0;
    cudaMemset(y_data, 0, n * sizeof(float64_t));
    cublasDaxpy_v2(handle, (int) n, &alpha, x_data, (int) x_stride, y_data, (int) y_stride);
    cudaDeviceSynchronize();
}

void cu_negation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_negation_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_negation_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_rectified_linear_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        float32_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float32_t) 0.0; 
    }
}

extern "C" static void cu_rectified_linear_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        float64_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float64_t) 0.0; 
    }
}

extern "C" void cu_rectified_linear(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_rectified_linear_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cu_rectified_linear_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

extern "C" void cu_addition(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    float32_t alpha_32 = 1.0;
    float64_t alpha_64 = 1.0;
    switch (datatype)
    {
    case FLOAT32:
        cublasScopy_v2(handle, (int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) z_data)[z_offset], (int) z_stride); 
        cudaDeviceSynchronize();
        cublasSaxpy_v2(handle, (int) n, &alpha_32, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        cudaDeviceSynchronize();
        break;
    case FLOAT64:
        cublasDcopy_v2(handle, (int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        cudaDeviceSynchronize();
        cublasDaxpy_v2(handle, (int) n, &alpha_64, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }
}

extern "C" void cu_subtraction(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    float32_t alpha_32 = -1.0;
    float64_t alpha_64 = -1.0;
    switch (datatype)
    {
    case FLOAT32:
        cublasScopy_v2(handle, (int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) z_data)[z_offset], (int) z_stride); 
        cudaDeviceSynchronize();
        cublasSaxpy_v2(handle, (int) n, &alpha_32, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        cudaDeviceSynchronize();
        break;
    case FLOAT64:
        cublasDcopy_v2(handle, (int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        cudaDeviceSynchronize();
        cublasDaxpy_v2(handle, (int) n, &alpha_64, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }
}

extern "C" static void cu_multiplication_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] * y_data[i * y_stride];
    }
}

extern "C" static void cu_multiplication_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] * y_data[i * y_stride];
    }
}

extern "C" void cu_multiplication(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_multiplication_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cu_multiplication_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_division_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] / y_data[i * y_stride];
    }
}

extern "C" static void cu_division_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] / y_data[i * y_stride];
    }
}

extern "C" void cu_division(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_division_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cu_division_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_power_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = powf(x_data[i * x_stride], y_data[i * y_stride]);
    }
}

extern "C" static void cu_power_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = pow(x_data[i * x_stride], y_data[i * y_stride]);
    }
}

extern "C" void cu_power(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_power_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cu_power_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_compare_equal_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] == y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

extern "C" static void cu_compare_equal_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] == y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

extern "C" void cu_compare_equal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_compare_equal_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cu_compare_equal_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_compare_greater_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

extern "C" static void cu_compare_greater_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

extern "C" void cu_compare_greater(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_compare_greater_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cu_compare_greater_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

extern "C" void cu_matrix_multiplication(datatype_t datatype, uint64_t m, uint64_t k, uint64_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, uint64_t x_offset, const void *y_data, uint64_t y_offset, void *z_data, uint64_t z_offset)
{
    float32_t beta_32 = 0.0;
    float32_t alpha_32 = 1.0;
    float64_t beta_64 = 0.0;
    float64_t alpha_64 = 1.0;
    switch (datatype)
    {
    case FLOAT32:
        cublasSgemm_v2(handle, (y_transpose) ? CUBLAS_OP_N : CUBLAS_OP_T, (x_transpose) ? CUBLAS_OP_N : CUBLAS_OP_T, n, m, k, &alpha_32,
                       &((float32_t *) y_data)[y_offset], n, &((float32_t *) x_data)[x_offset], k, &beta_32, &((float32_t *) z_data)[z_offset], n);
        cudaDeviceSynchronize();
        break;
    case FLOAT64:
        cublasDgemm_v2(handle, (y_transpose) ? CUBLAS_OP_N : CUBLAS_OP_T, (x_transpose) ? CUBLAS_OP_N : CUBLAS_OP_T, n, m, k, &alpha_64,
                       &((float64_t *) y_data)[y_offset], n, &((float64_t *) x_data)[x_offset], k, &beta_64, &((float64_t *) z_data)[z_offset], n);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }
}

extern "C" static void cu_summation_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t temp = 1.0;
    cublasSdot_v2(handle, n, x_data, x_stride, &temp, (int) 0, y_data);
}

extern "C" static void cu_summation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t temp = 1.0;
    cublasDdot_v2(handle, n, x_data, x_stride, &temp, (int) 0, y_data);
}

extern "C" void cu_summation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_summation_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);
        break;
    case FLOAT64:
        cu_summation_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);
        break;
    default:
        break;
    }
}

extern "C" static void cu_maximum_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t maximum = *x_data;
    for (int i = 1; i < n; ++i)
    {
        float32_t candidate = x_data[i * x_stride];
        if (maximum < candidate)
        {
            maximum = candidate;
        }
    }
    *y_data = maximum;
}

extern "C" static void cu_maximum_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t maximum = *x_data;
    for (int i = 1; i < n; ++i)
    {
        float64_t candidate = x_data[i * x_stride];
        if (maximum < candidate)
        {
            maximum = candidate;
        }
    }
    *y_data = maximum;
}

extern "C" void cu_maximum(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_maximum_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);
        break;
    case FLOAT64:
        cu_maximum_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);
        break;
    default:
        break;
    }
}
