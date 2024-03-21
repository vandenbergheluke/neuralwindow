/**@file cu_runtime.cu
 * @brief Implementation of low level matrix operations using CUDA kernels and
 * CuBLAS.
 */

#include "magma_types.h"
#include "magmablas_s.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
extern "C" {
    #include <cu_runtime.h>
}
#include <magma_v2.h>

#ifndef SYNCHRONOUS
#define SYNCHRONOUS 1
#endif

#define EPSILON 1e-7

#define NW_WARP_SIZE 32

#define NW_NUM_STREAMS 16

#define NW_CHUNK_SIZE 64

// CUDA defns.
static cublasHandle_t cublas_handle = NULL;
static cusparseHandle_t cusparse_handle = NULL;
static cudaStream_t cuda_stream[NW_NUM_STREAMS] = {0};

// MAGMA defns.
static magma_queue_t m_queue[NW_NUM_STREAMS] = {0};

extern "C" nw_error_t *cu_create_context(void)
{
    cublasStatus_t cublasStatus = cublasCreate_v2(&cublas_handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create cublas context."), NULL);
    }

    cusparseStatus_t cusparseStatus = cusparseCreate(&cusparse_handle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create cusparse context."), NULL);
    }

    for (int i = 0; i < NW_NUM_STREAMS; ++i)
    {
        cudaStreamCreate(&cuda_stream[i]);
    }

    magma_int_t error = magma_init();
    if (error != MAGMA_SUCCESS) {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to initialize MAGMA.", magma_strerror(error)), NULL);
    }

    magma_device_t m_device;
    magma_getdevice(&m_device);

    for (int i = 0; i < NW_NUM_STREAMS; ++i)
    {
        magma_queue_create_from_cuda(m_device, cuda_stream[i], cublas_handle, cusparse_handle, &m_queue[i]);
    }

    return NULL;
}

extern "C" void cu_destroy_context(void)
{
    // Automatically synchronizes the device.
    cublasDestroy_v2(cublas_handle);
    cusparseDestroy(cusparse_handle);
    for (int i = 0; i < NW_NUM_STREAMS; ++i)
    {
        cudaStreamDestroy(cuda_stream[i]);

        magma_queue_destroy(m_queue[i]);
    }
    magma_finalize();
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

extern "C" void cu_synchronize(void)
{
    cudaDeviceSynchronize();
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_exponential_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_exponential_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_logarithm_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;

    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_logarithm_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_sine_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_sine_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_cosine_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_cosine_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_square_root_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_square_root_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_reciprocal_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_reciprocal_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    default:
        break;
    }
}

extern "C" void cu_copy(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    // TODO: We don't have any way to measure the performance of these functions
    // yet. We'll use MAGMA for now.
    switch (datatype)
    {
    case FLOAT32:
#if 1
        cudaDeviceSynchronize();
        magma_scopy((magma_int_t) n, &((magmaFloat_const_ptr) x_data)[x_offset], (magma_int_t) x_stride, &((magmaFloat_ptr) y_data)[y_offset], (magma_int_t) y_stride, m_queue[0]);
        magma_queue_sync(m_queue[0]);
#else
        cublasScopy_v2(cublas_handle, (int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
#if 1
        cudaDeviceSynchronize();
        magma_dcopy((magma_int_t) n, &((magmaDouble_const_ptr) x_data)[x_offset], (magma_int_t) x_stride, &((magmaDouble_ptr) y_data)[y_offset], (magma_int_t) y_stride, m_queue[0]);
        magma_queue_sync(m_queue[0]);
#else
        cublasDcopy_v2(cublas_handle, (int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_negation_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_negation_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_rectified_linear_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_rectified_linear_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
        y_data[i * y_stride] = (float32_t) 1.0 / ((float32_t) 1.0 + expf(-x));
    }
}

__global__ static void cu_sigmoid_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        float64_t x = x_data[i * x_stride];
        y_data[i * y_stride] = (float64_t) 1.0 / ((float64_t) 1.0 + exp(-x));
    }
}

extern "C" void cu_sigmoid(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    int block_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_sigmoid_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_sigmoid_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    float alpha = 1.0;
    cudaDeviceSynchronize();
    magma_scopy((magma_int_t) n, (magmaFloat_const_ptr) x_data, (magma_int_t) x_stride, (magmaFloat_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
    magma_saxpy((magma_int_t) n, alpha, (magmaFloat_const_ptr) y_data, (magma_int_t) y_stride, (magmaFloat_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
}

extern "C" static void cu_addition_float64(int n,
                                           const float64_t *x_data,
                                           int x_stride,
                                           const float64_t *y_data,
                                           int y_stride,
                                           double *z_data,
                                           float64_t z_stride)
{
    double alpha = 1.0;
    cudaDeviceSynchronize();
    magma_dcopy((magma_int_t) n, (magmaDouble_const_ptr) x_data, (magma_int_t) x_stride, (magmaDouble_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
    magma_daxpy((magma_int_t) n, alpha, (magmaDouble_const_ptr) y_data, (magma_int_t) y_stride, (magmaDouble_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
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
    float alpha = -1.0;
    cudaDeviceSynchronize();
    magma_scopy((magma_int_t) n, (magmaFloat_const_ptr) x_data, (magma_int_t) x_stride, (magmaFloat_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
    magma_saxpy((magma_int_t) n, alpha, (magmaFloat_const_ptr) y_data, (magma_int_t) y_stride, (magmaFloat_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
}

extern "C" static void cu_subtraction_float64(int n,
                                              const float64_t *x_data,
                                              int x_stride,
                                              const float64_t *y_data,
                                              int y_stride,
                                              float64_t *z_data,
                                              int z_stride)
{
    double alpha = -1.0;
    cudaDeviceSynchronize();
    magma_dcopy((magma_int_t) n, (magmaDouble_const_ptr) x_data, (magma_int_t) x_stride, (magmaDouble_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
    magma_daxpy((magma_int_t) n, alpha, (magmaDouble_const_ptr) y_data, (magma_int_t) y_stride, (magmaDouble_ptr) z_data, (magma_int_t) z_stride, m_queue[0]);
    magma_queue_sync(m_queue[0]);
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_multiplication_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float32_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_multiplication_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float64_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_division_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float32_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_division_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float64_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_power_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float32_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_power_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float64_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_compare_equal_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float32_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_compare_equal_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float64_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_compare_greater_float32<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float32_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float32_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float32_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        for (int i = 0, j = 0; i < n; i += NW_CHUNK_SIZE, ++j)
        {
            cu_compare_greater_float64<<<grid_size,
                    block_size,
                    0,
                    cuda_stream[j % NW_NUM_STREAMS]>>>(
                        MIN(NW_CHUNK_SIZE, (int) n - i),
                        &((float64_t *) x_data)[x_offset + (i * x_stride)],
                        (int) x_stride,
                        &((float64_t *) y_data)[y_offset + (i * y_stride)],
                        (int) y_stride,
                        &((float64_t *) z_data)[z_offset + (i * z_stride)],
                        (int) z_stride);
        }

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
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
    float alpha = 1.0;
    float beta = 0.0;
    cudaDeviceSynchronize();
    magma_sgemm(x_transpose ? MagmaTrans : MagmaNoTrans,
            y_transpose ? MagmaTrans : MagmaNoTrans,
            n, m, k, alpha, (magmaFloat_const_ptr) y_data,
            n, (magmaFloat_const_ptr) x_data, k, beta,
            (magmaFloat_ptr) z_data, n, m_queue[0]);
    magma_queue_sync(m_queue[0]);
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
    double alpha = 1.0;
    double beta = 0.0;
    cudaDeviceSynchronize();
    magma_dgemm(x_transpose ? MagmaTrans : MagmaNoTrans,
            y_transpose ? MagmaTrans : MagmaNoTrans,
            n, m, k, alpha, (magmaDouble_const_ptr) y_data,
            n, (magmaDouble_const_ptr) x_data, k, beta,
            (magmaDouble_ptr) z_data, n, m_queue[0]);
    magma_queue_sync(m_queue[0]);
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
    // This one is a tossup with cublas in terms of performance, and we have a
    // bit of a blindspot when it comes to smaller matrices, but we'll use
    // MAGMA for now.
    float32_t *temp;
    cudaMallocManaged((void **) &temp, sizeof(float32_t));
    *temp = (float32_t) 1.0;
    cudaDeviceSynchronize();
    cublasSdot(cublas_handle, n, x_data, x_stride, temp, 0, y_data);
    cudaDeviceSynchronize();
    cudaFree(temp);
}

extern "C" static void cu_summation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    // This one is a tossup with cublas in terms of performance, and we have a
    // bit of a blindspot when it comes to smaller matrices, but we'll use
    // MAGMA for now.
    float64_t *temp;
    cudaMallocManaged((void **) &temp, sizeof(float64_t));
    *temp = (float64_t) 1.0;
    cudaDeviceSynchronize();
    cublasDdot(cublas_handle, n, x_data, x_stride, temp, 0, y_data);
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

// Since atomicMAX doesn't normally support float
__device__ static float32_t atomicMAX(float32_t *address, float32_t val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float64_t atomicMAX(float64_t *address, float64_t val)
{
    long long unsigned* address_as_ull = (long long unsigned*) address;
    long long unsigned old = *address_as_ull;
    long long unsigned assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ static void cu_maximum_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    __shared__ float32_t current_maximum;
    current_maximum = *x_data;
    __syncthreads();
    int i = (blockDim.x * blockIdx.x) + threadIdx.x + 1;
    if (i < n)
    {
        atomicMAX(&current_maximum, x_data[i * x_stride]);
    }
    __syncthreads();
    *y_data = current_maximum;
}

__global__ static void cu_maximum_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    __shared__ float64_t current_maximum;
    current_maximum = *x_data;
    __syncthreads();
    int i = (blockDim.x * blockIdx.x) + threadIdx.x + 1;
    if (i < n)
    {
        atomicMAX(&current_maximum, x_data[i * x_stride]);
    }
    __syncthreads();
    *y_data = current_maximum;
}

extern "C" void cu_maximum(datatype_t datatype, int64_t n, const void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_offset)
{
    int block_size;
    int grid_size;

    switch (datatype)
    {
    case FLOAT32:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        // TODO: Figure out a good way to dispatch in chunks
        cudaDeviceSynchronize();
        cu_maximum_float32<<<grid_size, block_size>>>((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    case FLOAT64:
        block_size = NW_WARP_SIZE * 24;

        grid_size = (MIN(NW_CHUNK_SIZE, n) + block_size - 1) / block_size;

        // TODO: Figure out a good way to dispatch in chunks
        cudaDeviceSynchronize();
        cu_maximum_float64<<<grid_size, block_size>>>((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);

#if SYNCHRONOUS
        cudaDeviceSynchronize();
#endif
        break;
    default:
        break;
    }
}
