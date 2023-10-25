/**@file cu_runtime.cu
 * @brief Implementation of low level matrix operations using CUDA kernels and
 * CuBLAS.
 */

#include <cuda_runtime.h>
#include <cublas.h>
extern "C" {
    #include <cu_runtime.h>
}

#define EPSILON 1e-7

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
    // Automatically synchronizes the device.
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

__global__ static void cu_exponential_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = expf(x_data[i * x_stride]);
    }
}

__global__ static void cu_exponential_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = exp(x_data[i * x_stride]);
    }
}

extern "C" void cu_exponential(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    // CUDA devs want us using ints here for minor optimization purposes, and
    // presumably because we know we're not going to overflow.
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        // I want to move this code into cu_context_create and store the grid
        // and block size somewhere, but presumably
        // cudaOccupancyMaxPotentialBlockSize is compile time so we're only
        // losing time on the division, and darknet does something completely
        // different so I think it's best to try to understand that before doing
        // any major restructuring.
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_exponential_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_exponential_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_exponential_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_exponential_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_logarithm_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = logf(x_data[i * x_stride]);
    }
}

__global__ static void cu_logarithm_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = log(x_data[i * x_stride]);
    }
}

extern "C" void cu_logarithm(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_logarithm_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_logarithm_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;

    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_logarithm_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_logarithm_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_sine_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = sinf(x_data[i * x_stride]);
    }
}

__global__ static void cu_sine_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = sin(x_data[i * x_stride]);
    }
}

extern "C" void cu_sine(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_sine_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_sine_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_sine_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_sine_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_cosine_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = cosf(x_data[i * x_stride]);
    }
}

__global__ static void cu_cosine_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = cos(x_data[i * x_stride]);
    }
}

extern "C" void cu_cosine(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_cosine_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_cosine_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_cosine_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        cu_cosine_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_square_root_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = sqrtf(x_data[i * x_stride]);
    }
}

__global__ static void cu_square_root_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = sqrt(x_data[i * x_stride]);
    }
}

extern "C" void cu_square_root(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_square_root_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_square_root_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_square_root_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_square_root_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_reciprocal_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = 1. / x_data[i * x_stride];
    }
}

__global__ static void cu_reciprocal_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = 1. / x_data[i * x_stride];
    }
}

extern "C" void cu_reciprocal(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_reciprocal_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_reciprocal_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_reciprocal_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_reciprocal_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

extern "C" void cu_copy(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
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

__global__ static void cu_negation_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = -x_data[i * x_stride];
    }
}

__global__ static void cu_negation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y_data[i * y_stride] = -x_data[i * x_stride];
    }
}

void cu_negation(datatype_t datatype,
                 int64_t n,
                 const void *x_data,
                 int64_t x_stride,
                 int64_t x_offset,
                 void *y_data,
                 int64_t y_stride,
                 int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_negation_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_negation_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_negation_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_negation_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_rectified_linear_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        float32_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float32_t) 0.0;
    }
}

__global__ static void cu_rectified_linear_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        float64_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float64_t) 0.0;
    }
}

extern "C" void cu_rectified_linear(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_rectified_linear_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_rectified_linear_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_rectified_linear_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_rectified_linear_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_sigmoid_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        float32_t x = x_data[i * x_stride];
        if (x >= 0)
        {
            y_data[i * y_stride] = (float32_t) 1.0 / ((float32_t) 1.0 + expf(-x)); 
        }
        else
        {
            y_data[i * y_stride] = expf(x) / ((float32_t) 1.0 + expf(x)); 
        }
    }
}

__global__ static void cu_sigmoid_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        float64_t x = x_data[i * x_stride];
        if (x >= 0)
        {
            y_data[i * y_stride] = (float64_t) 1.0 / ((float64_t) 1.0 + exp(-x)); 
        }
        else
        {
            y_data[i * y_stride] = exp(x) / ((float64_t) 1.0 + exp(x)); 
        }
    }
}

extern "C" void cu_sigmoid(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_sigmoid_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_sigmoid_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_sigmoid_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_sigmoid_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);

        break;
    default:
        break;
    }
}

extern "C" static void cu_addition_float32(int n,
                                           const float32_t *x_data,
                                           int x_stride,
                                           const float32_t *y_data,
                                           int y_stride,
                                           float32_t *z_data,
                                           int z_stride)
{
    float32_t *alpha;
    cudaMallocManaged((void **) &alpha, sizeof(float32_t));
    *alpha = (float32_t) 1.0;
    cublasScopy_v2(handle, n, x_data, x_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cublasSaxpy_v2(handle, n, alpha, y_data, y_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cudaFree(alpha);

}

extern "C" static void cu_addition_float64(int n,
                                           const float64_t *x_data,
                                           int x_stride,
                                           const float64_t *y_data,
                                           int y_stride,
                                           double *z_data,
                                           float64_t z_stride)
{
    float64_t *alpha;
    cudaMallocManaged((void **) &alpha, sizeof(float64_t));
    *alpha = (float64_t) 1.0;
    cublasDcopy_v2(handle, n, x_data, x_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cublasDaxpy_v2(handle, n, alpha, y_data, y_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cudaFree(alpha);

}

extern "C" void cu_addition(datatype_t datatype,
                            int64_t n,
                            const void *x_data,
                            int64_t x_stride,
                            int64_t x_offset,
                            const void *y_data,
                            int64_t y_stride,
                            int64_t y_offset,
                            void *z_data,
                            int64_t z_stride,
                            int64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_addition_float32((int) n, 
                            &((float32_t *) x_data)[x_offset], 
                            (int) x_stride,
                            &((float32_t *) y_data)[y_offset],
                            (int) y_stride,
                            &((float32_t *) z_data)[z_offset],
                            (int) z_stride);
        break;
    case FLOAT64:
        cu_addition_float64((int) n, 
                            &((float64_t *) x_data)[x_offset], 
                            (int) x_stride,
                            &((float64_t *) y_data)[y_offset],
                            (int) y_stride,
                            &((float64_t *) z_data)[z_offset],
                            (int) z_stride);
        break;
    default:
        break;
    }
}

extern "C" static void cu_subtraction_float32(int n,
                                              const float32_t *x_data,
                                              int x_stride,
                                              const float32_t *y_data,
                                              int y_stride,
                                              float32_t *z_data,
                                              int z_stride)
{
    float32_t *alpha;
    cudaMallocManaged((void **) &alpha, sizeof(float32_t));
    *alpha = (float32_t) -1.0;
    cublasScopy_v2(handle, n, x_data, x_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cublasSaxpy_v2(handle, n, alpha, y_data, y_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cudaFree(alpha);
}

extern "C" static void cu_subtraction_float64(int n,
                                              const float64_t *x_data,
                                              int x_stride,
                                              const float64_t *y_data,
                                              int y_stride,
                                              float64_t *z_data,
                                              int z_stride)
{
    float64_t *alpha;
    cudaMallocManaged((void **) &alpha, sizeof(float64_t));
    *alpha = (float64_t) -1.0;
    cublasDcopy_v2(handle, n, x_data, x_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cublasDaxpy_v2(handle, n, alpha, y_data, y_stride, z_data, z_stride);
    cudaDeviceSynchronize();
    cudaFree(alpha);

}

extern "C" void cu_subtraction(datatype_t datatype,
                               int64_t n,
                               const void *x_data,
                               int64_t x_stride,
                               int64_t x_offset,
                               const void *y_data,
                               int64_t y_stride,
                               int64_t y_offset,
                               void *z_data,
                               int64_t z_stride,
                               int64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_subtraction_float32((int) n, 
                               &((float32_t *) x_data)[x_offset], 
                               (int) x_stride,
                               &((float32_t *) y_data)[y_offset],
                               (int) y_stride,
                               &((float32_t *) z_data)[z_offset],
                               (int) z_stride);
        break;
    case FLOAT64:
        cu_subtraction_float64((int) n, 
                               &((float64_t *) x_data)[x_offset], 
                               (int) x_stride,
                               &((float64_t *) y_data)[y_offset],
                               (int) y_stride,
                               &((float64_t *) z_data)[z_offset],
                               (int) z_stride);
        break;
    default:
        break;
    }
}

__global__ static void cu_multiplication_float32(int n,
                                                 const float32_t *x_data,
                                                 int x_stride,
                                                 const float32_t *y_data,
                                                 int y_stride,
                                                 float32_t *z_data,
                                                 int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = x_data[i * x_stride] * y_data[i * y_stride];
    }
}

__global__ static void cu_multiplication_float64(int n,
                                                 const float64_t *x_data,
                                                 int x_stride,
                                                 const float64_t *y_data,
                                                 int y_stride,
                                                 float64_t *z_data,
                                                 int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = x_data[i * x_stride] * y_data[i * y_stride];
    }
}

extern "C" void cu_multiplication(datatype_t datatype,
                                  int64_t n,
                                  const void *x_data,
                                  int64_t x_stride,
                                  int64_t x_offset,
                                  const void *y_data,
                                  int64_t y_stride,
                                  int64_t y_offset,
                                  void *z_data,
                                  int64_t z_stride,
                                  int64_t z_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_multiplication_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_multiplication_float32<<<grid_size, block_size>>>((int) n,
                                  &((float32_t *) x_data)[x_offset],
                                  (int) x_stride,
                                  &((float32_t *) y_data)[y_offset],
                                  (int) y_stride,
                                  &((float32_t *) z_data)[z_offset],
                                  (int) z_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_multiplication_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_multiplication_float64<<<grid_size, block_size>>>((int) n,
                                  &((float64_t *) x_data)[x_offset],
                                  (int) x_stride,
                                  &((float64_t *) y_data)[y_offset],
                                  (int) y_stride,
                                  &((float64_t *) z_data)[z_offset],
                                  (int) z_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_division_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = x_data[i * x_stride] / y_data[i * y_stride];
    }
}

__global__ static void cu_division_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = x_data[i * x_stride] / y_data[i * y_stride];
    }
}

extern "C" void cu_division(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, const void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_division_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_division_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_division_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_division_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_power_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = powf(x_data[i * x_stride], y_data[i * y_stride]);
    }
}

__global__ static void cu_power_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = pow(x_data[i * x_stride], y_data[i * y_stride]);
    }
}

extern "C" void cu_power(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, const void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_power_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_power_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_power_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_power_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_compare_equal_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    float32_t x, y;
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        x = x_data[i * x_stride];
        y = y_data[i * y_stride];
        z_data[i * z_stride] = fabsf(x - y) < EPSILON ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

__global__ static void cu_compare_equal_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    float64_t x, y;
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        x = x_data[i * x_stride];
        y = y_data[i * y_stride];
        z_data[i * z_stride] = fabs(x - y) < EPSILON ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

extern "C" void cu_compare_equal(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, const void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_compare_equal_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_compare_equal_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_compare_equal_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_compare_equal_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);

        break;
    default:
        break;
    }
}

__global__ static void cu_compare_greater_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

__global__ static void cu_compare_greater_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

extern "C" void cu_compare_greater(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, const void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_compare_greater_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_compare_greater_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_compare_greater_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_compare_greater_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);

        break;
    default:
        break;
    }
}

extern "C" void cu_matrix_multiplication_float32(datatype_t datatype,
                                                 int64_t m,
                                                 int64_t k,
                                                 int64_t n,
                                                 bool_t x_transpose,
                                                 bool_t y_transpose,
                                                 const float32_t *x_data,
                                                 const float32_t *y_data,
                                                 float32_t *z_data)
{
    float32_t *alpha;
    float32_t *beta;
    cudaMallocManaged((void **) &alpha, sizeof(float32_t));
    cudaMallocManaged((void **) &beta, sizeof(float32_t));
    *alpha = (float32_t) 1.0;
    *beta = (float32_t) 0.0;
    cublasSgemm_v2(handle,
                   y_transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                   x_transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                   n, m, k, alpha,
                   y_data, n, x_data, 
                   k, beta, z_data, n);
    cudaDeviceSynchronize();
    cudaFree(alpha);
    cudaFree(beta);
}

extern "C" void cu_matrix_multiplication_float64(datatype_t datatype,
                                                 int64_t m,
                                                 int64_t k,
                                                 int64_t n,
                                                 bool_t x_transpose,
                                                 bool_t y_transpose,
                                                 const float64_t *x_data,
                                                 const float64_t *y_data,
                                                 float64_t *z_data)
{
    float64_t *alpha;
    float64_t *beta;
    cudaMallocManaged((void **) &alpha, sizeof(float64_t));
    cudaMallocManaged((void **) &beta, sizeof(float64_t));
    *alpha = (float64_t) 1.0;
    *beta = (float64_t) 0.0;
    cublasDgemm_v2(handle,
                   y_transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                   x_transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                   n, m, k, alpha,
                   y_data, n, x_data, 
                   k, beta, z_data, n);
    cudaDeviceSynchronize();
    cudaFree(alpha);
    cudaFree(beta);
}

extern "C" void cu_matrix_multiplication(datatype_t datatype,
                                         int64_t m,
                                         int64_t k,
                                         int64_t n,
                                         bool_t x_transpose,
                                         bool_t y_transpose,
                                         const void *x_data,
                                         int64_t x_offset,
                                         const void *y_data,
                                         int64_t y_offset,
                                         void *z_data,
                                         int64_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cu_matrix_multiplication_float32(datatype,
                                         m,
                                         k,
                                         n,
                                         x_transpose,
                                         y_transpose,
                                         &((float32_t *) x_data)[x_offset],
                                         &((float32_t *) y_data)[y_offset],
                                         &((float32_t *) z_data)[z_offset]);
        break;
    case FLOAT64:
        cu_matrix_multiplication_float64(datatype,
                                         m,
                                         k,
                                         n,
                                         x_transpose,
                                         y_transpose,
                                         &((float64_t *) x_data)[x_offset],
                                         &((float64_t *) y_data)[y_offset],
                                         &((float64_t *) z_data)[z_offset]);
        break;
    default:
        break;
    }
}

extern "C" static void cu_summation_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t *temp;
    cudaMallocManaged((void **) &temp, sizeof(float32_t));
    *temp = (float32_t) 1.0;
    cublasSdot_v2(handle, n, x_data, x_stride, temp, (int) 0, y_data);
    cudaDeviceSynchronize();
    cudaFree(temp);
}

extern "C" static void cu_summation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t *temp;
    cudaMallocManaged((void **) &temp, sizeof(float64_t));
    *temp = (float64_t) 1.0;
    cublasDdot_v2(handle, n, x_data, x_stride, temp, (int) 0, y_data);
    cudaDeviceSynchronize();
    cudaFree(temp);
}

extern "C" void cu_summation(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_offset)
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

__global__ static void cu_maximum_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t maximum = *x_data;
    int i = (blockDim.x * blockIdx.x) + threadIdx.x + 1;
    if (i < n)
    {
        float32_t candidate = x_data[i * x_stride];
        if (maximum < candidate)
        {
            maximum = candidate;
        }
    }
    *y_data = maximum;
}

__global__ static void cu_maximum_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t maximum = *x_data;
    int i = (blockDim.x * blockIdx.x) + threadIdx.x + 1;
    if (i < n)
    {
        float64_t candidate = x_data[i * x_stride];
        if (maximum < candidate)
        {
            maximum = candidate;
        }
    }
    *y_data = maximum;
}

extern "C" void cu_maximum(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_offset)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_maximum_float32, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_maximum_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);

        break;
    case FLOAT64:
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                cu_maximum_float64, 0, 0);

        if (block_size == 0)
        {
            block_size = 32;
        }
        grid_size = (n + block_size - 1) / block_size;

        cu_maximum_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);

        break;
    default:
        break;
    }
}
