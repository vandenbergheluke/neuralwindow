#include <openblas_runtime.h>
#include <cblas.h>
#include <math.h>

nw_error_t *openblas_memory_allocate(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    *pp = malloc(size);
    if (*pp == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    return NULL;
}

void openblas_memory_free(void *p)
{
    free(p);
}

static void openblas_exponential_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = expf(x_data[i * x_stride]); 
    }
}

static void openblas_exponential_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = exp(x_data[i * x_stride]); 
    }
}

void openblas_exponential(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_exponential_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_exponential_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void openblas_logarithm_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = logf(x_data[i * x_stride]); 
    }
}

static void openblas_logarithm_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = log(x_data[i * x_stride]); 
    }
}

void openblas_logarithm(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_logarithm_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_logarithm_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void openblas_sine_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sinf(x_data[i * x_stride]); 
    }
}

static void openblas_sine_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sin(x_data[i * x_stride]); 
    }
}

void openblas_sine(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_sine_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_sine_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void openblas_cosine_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = cosf(x_data[i * x_stride]); 
    }
}

static void openblas_cosine_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = cos(x_data[i * x_stride]); 
    }
}

void openblas_cosine(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_cosine_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_cosine_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void openblas_square_root_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sqrtf(x_data[i * x_stride]); 
    }
}

static void openblas_square_root_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = sqrt(x_data[i * x_stride]); 
    }
}

void openblas_square_root(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_square_root_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_square_root_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void openblas_reciprocal_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = 1. / x_data[i * x_stride]; 
    }
}

static void openblas_reciprocal_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        y_data[i * y_stride] = 1. / x_data[i * x_stride]; 
    }
}

void openblas_reciprocal(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_reciprocal_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_reciprocal_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void openblas_copy(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_scopy((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cblas_dcopy((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void openblas_negation(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_saxpby((int) n, (float32_t) -1.0, &((float32_t *) x_data)[x_offset], (int) x_stride, (float32_t) 0.0, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cblas_daxpby((int) n, (float64_t) -1.0, &((float64_t *) x_data)[x_offset], (int) x_stride, (float64_t) 0.0, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void openblas_rectified_linear_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        float32_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float32_t) 0.0; 
    }
}

static void openblas_rectified_linear_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; ++i)
    {
        float64_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float64_t) 0.0; 
    }
}

void openblas_rectified_linear(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_rectified_linear_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        openblas_rectified_linear_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void openblas_addition(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_scopy((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) z_data)[z_offset], (int) z_stride); 
        cblas_saxpy((int) n, 1.0, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cblas_dcopy((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        cblas_daxpy((int) n, 1.0, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void openblas_subtraction(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_scopy((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) z_data)[z_offset], (int) z_stride); 
        cblas_saxpy((int) n, -1.0, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        cblas_dcopy((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        cblas_daxpy((int) n, -1.0, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void openblas_multiplication_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] * y_data[i * y_stride];
    }
}

static void openblas_multiplication_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] * y_data[i * y_stride];
    }
}

void openblas_multiplication(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_multiplication_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        openblas_multiplication_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void openblas_division_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] / y_data[i * y_stride];
    }
}

static void openblas_division_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = x_data[i * x_stride] / y_data[i * y_stride];
    }
}

void openblas_division(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_division_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        openblas_division_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void openblas_power_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = powf(x_data[i * x_stride], y_data[i * y_stride]);
    }
}

static void openblas_power_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = pow(x_data[i * x_stride], y_data[i * y_stride]);
    }
}

void openblas_power(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_power_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        openblas_power_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void openblas_compare_equal_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] == y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

static void openblas_compare_equal_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] == y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

void openblas_compare_equal(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_compare_equal_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        openblas_compare_equal_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void openblas_compare_greater_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

static void openblas_compare_greater_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; ++i)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

void openblas_compare_greater(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_compare_greater_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        openblas_compare_greater_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void openblas_matrix_multiplication(datatype_t datatype, uint32_t m, uint32_t k, uint32_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, uint32_t x_offset, const void *y_data, uint32_t y_offset, void *z_data, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_sgemm(CblasRowMajor, (x_transpose) ? CblasNoTrans: CblasTrans, (y_transpose) ? CblasNoTrans : CblasTrans, (int) m, (int) n, (int) k, 1.0,
                    &((float32_t *) x_data)[x_offset], (int) m, &((float32_t *) y_data)[y_offset], (int) k, 0.0, &((float32_t *) z_data)[z_offset], (int) m);
        break;
    case FLOAT64:
        cblas_dgemm(CblasRowMajor, (x_transpose) ? CblasNoTrans: CblasTrans, (y_transpose) ? CblasNoTrans : CblasTrans, (int) m, (int) n, (int) k, 1.0, 
                    &((float64_t *) x_data)[x_offset], (int) m, &((float64_t *) y_data)[y_offset], (int) k, 0.0, &((float64_t *) z_data)[z_offset], (int) m);
        break;
    default:
        break;
    }
}

static void openblas_summation_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t temp = 1.0;
    *y_data = cblas_sdot(n, x_data, x_stride, &temp, (int) 1);
}

static void openblas_summation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t temp = 1.0;
    *y_data = cblas_ddot(n, x_data, x_stride, &temp, (int) 1);
}

void openblas_summation(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_summation_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);
        break;
    case FLOAT64:
        openblas_summation_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);
        break;
    default:
        break;
    }
}

static void openblas_maximum_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
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

static void openblas_maximum_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
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

void openblas_maximum(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        openblas_maximum_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);
        break;
    case FLOAT64:
        openblas_maximum_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);
        break;
    default:
        break;
    }
}
