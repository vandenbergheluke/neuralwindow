#include <runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>
#ifndef CPU_ONLY
#include <cu_runtime.h>
#endif
#include <random.h>

nw_error_t *runtime_create_context(runtime_t runtime)
{
    nw_error_t *error = NULL;

    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        error = openblas_create_context();
        break;
    case MKL_RUNTIME:
        error = mkl_create_context();
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        error = cu_create_context();
        break;
#endif
    default:
        error = ERROR(ERROR_RUNTIME, string_create("unknown runtime %d.", (int) runtime), NULL);
        break;
    }
    
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create context for runtime %s.", runtime_string(runtime)), error);
    }

    return NULL;
}

void runtime_destroy_context(runtime_t runtime)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_destroy_context();
        break;
#endif
    default:
        break;
    }
}

nw_error_t *runtime_malloc(void **data, void **ddata, int64_t n, datatype_t datatype, runtime_t runtime)
{
    if (!n)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("cannot allocate 0 bytes."), NULL);
    }

    nw_error_t *error = NULL;
    size_t size = n * datatype_size(datatype);

    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        error = openblas_memory_allocate(ddata, size);
        *data = *ddata;
        break;
    case MKL_RUNTIME:
        error = mkl_memory_allocate(ddata, size);
        *data = *ddata;
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        error = cu_memory_allocate(ddata, size);
        *data = malloc(size);
        break;
#endif
    default:
        error = ERROR(ERROR_RUNTIME, string_create("unknown runtime %d.", (int) runtime), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes for runtime %s.", size, runtime_string(runtime)), error);
    }
    
    return NULL;
}

void runtime_free(void *data, void *ddata, runtime_t runtime)
{
    if (data)
    {
        switch (runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_memory_free(ddata);
            break;
        case MKL_RUNTIME:
            mkl_memory_free(ddata);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_memory_free(ddata);
            free(data);
            break;
#endif
        default:
            break;
        }
    }

}

void runtime_dev_to_cpu(void *data, void *ddata, int64_t n, datatype_t datatype, runtime_t runtime)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_dev_to_cpu(data, ddata, n * datatype_size(datatype));
        break;
#endif
    default:
        break;
    }
}

void runtime_cpu_to_dev(void *data, void *ddata, int64_t n, datatype_t datatype, runtime_t runtime)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_cpu_to_dev(data, ddata, n * datatype_size(datatype));
        break;
#endif
    default:
        break;
    }
}

void runtime_unary(unary_operation_type_t unary_operation_type, runtime_t runtime, datatype_t datatype, int64_t n, 
                   void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        switch (unary_operation_type)
        {
        case EXPONENTIAL_OPERATION:
            openblas_exponential(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case LOGARITHM_OPERATION:
            openblas_logarithm(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SINE_OPERATION:
            openblas_sine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case COSINE_OPERATION:
            openblas_cosine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SQUARE_ROOT_OPERATION:
            openblas_square_root(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case CONTIGUOUS_OPERATION:
            openblas_copy(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case NEGATION_OPERATION:
            openblas_negation(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RECTIFIED_LINEAR_OPERATION:
            openblas_rectified_linear(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SIGMOID_OPERATION:
            openblas_sigmoid(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RECIPROCAL_OPERATION:
            openblas_reciprocal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        default:
            break;
        }
        break;
    case MKL_RUNTIME:
        switch (unary_operation_type)
        {
        case EXPONENTIAL_OPERATION:
            mkl_exponential(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case LOGARITHM_OPERATION:
            mkl_logarithm(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SINE_OPERATION:
            mkl_sine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case COSINE_OPERATION:
            mkl_cosine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SQUARE_ROOT_OPERATION:
            mkl_square_root(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case CONTIGUOUS_OPERATION:
            mkl_copy(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case NEGATION_OPERATION:
            mkl_negation(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RECTIFIED_LINEAR_OPERATION:
            mkl_rectified_linear(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SIGMOID_OPERATION:
            mkl_sigmoid(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RECIPROCAL_OPERATION:
            mkl_reciprocal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        default:
            break;
        }
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        switch (unary_operation_type)
        {
        case EXPONENTIAL_OPERATION:
            cu_exponential(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case LOGARITHM_OPERATION:
            cu_logarithm(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SINE_OPERATION:
            cu_sine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case COSINE_OPERATION:
            cu_cosine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SQUARE_ROOT_OPERATION:
            cu_square_root(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case CONTIGUOUS_OPERATION:
            cu_copy(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case NEGATION_OPERATION:
            cu_negation(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RECTIFIED_LINEAR_OPERATION:
            cu_rectified_linear(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case SIGMOID_OPERATION:
            cu_sigmoid(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RECIPROCAL_OPERATION:
            cu_reciprocal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
}

void runtime_binary_elementwise(binary_operation_type_t binary_operation_type, runtime_t runtime, datatype_t datatype, int64_t n,
                                void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset,
                                void *z_data, int64_t z_stride, int64_t z_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        switch (binary_operation_type)
        {
        case ADDITION_OPERATION:
            openblas_addition(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case SUBTRACTION_OPERATION:
            openblas_subtraction(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case MULTIPLICATION_OPERATION:
            openblas_multiplication(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case DIVISION_OPERATION:
            openblas_division(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case POWER_OPERATION:  
            openblas_power(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case COMPARE_EQUAL_OPERATION:
            openblas_compare_equal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case COMPARE_GREATER_OPERATION:
            openblas_compare_greater(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        default:
            break;
        }
        break;
    case MKL_RUNTIME:
        switch (binary_operation_type)
        {
        case ADDITION_OPERATION:
            mkl_addition(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case SUBTRACTION_OPERATION:
            mkl_subtraction(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case MULTIPLICATION_OPERATION:
            mkl_multiplication(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case DIVISION_OPERATION:
            mkl_division(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case POWER_OPERATION:  
            mkl_power(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case COMPARE_EQUAL_OPERATION:
            mkl_compare_equal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case COMPARE_GREATER_OPERATION:
            mkl_compare_greater(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        default:
            break;
        }
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        switch (binary_operation_type)
        {
        case ADDITION_OPERATION:
            cu_addition(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case SUBTRACTION_OPERATION:
            cu_subtraction(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case MULTIPLICATION_OPERATION:
            cu_multiplication(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case DIVISION_OPERATION:
            cu_division(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case POWER_OPERATION:  
            cu_power(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case COMPARE_EQUAL_OPERATION:
            cu_compare_equal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case COMPARE_GREATER_OPERATION:
            cu_compare_greater(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
}

void runtime_matrix_multiplication(runtime_t runtime, datatype_t datatype, int64_t m, int64_t k, int64_t n, bool_t x_transpose, bool_t y_transpose,
                                   void *x_data, int64_t x_offset, void *y_data, int64_t y_offset, void *z_data, int64_t z_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_matrix_multiplication(datatype, m, k, n, x_transpose, y_transpose, x_data, x_offset, y_data, y_offset, z_data, z_offset);
        break;
    case MKL_RUNTIME:
        mkl_matrix_multiplication(datatype, m, k, n, x_transpose, y_transpose, x_data, x_offset, y_data, y_offset, z_data, z_offset);
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_matrix_multiplication(datatype, m, k, n, x_transpose, y_transpose, x_data, x_offset, y_data, y_offset, z_data, z_offset);
        break;
#endif
    default:
        break;
    }
}

void runtime_where(runtime_t runtime, datatype_t datatype, int64_t n,
                   void *w_data, int64_t w_stride, int64_t w_offset, void *x_data, int64_t x_stride, int64_t x_offset, 
                   void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
    #ifndef CPU_ONLY
    case CU_RUNTIME:
    #endif
        for (int64_t i = 0; i < n; ++i)
        {
            switch (datatype)
            {
            case FLOAT32:
                ((float32_t *) z_data)[z_offset + i * z_stride] = fabsf(((float32_t *) w_data)[w_offset + i * w_stride]) > EPSILON ? 
                                                                ((float32_t *) x_data)[x_offset + i * x_stride] : ((float32_t *) y_data)[y_offset + i * y_stride];
                break;
            case FLOAT64:
                ((float64_t *) z_data)[z_offset + i * z_stride] = fabs(((float64_t *) w_data)[w_offset + i * w_stride]) > EPSILON ? 
                                                                ((float64_t *) x_data)[x_offset + i * x_stride] : ((float64_t *) y_data)[y_offset + i * y_stride];
                break;
            default:
                break;
            }
        }
    default:
        break;
    }
}

void runtime_ternary(ternary_operation_type_t ternary_operation_type, runtime_t runtime, datatype_t datatype, int64_t n,
                     void *w_data, int64_t w_stride, int64_t w_offset, void *x_data, int64_t x_stride, int64_t x_offset, 
                     void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset)
{
    switch (ternary_operation_type)
    {
    case WHERE_OPERATION:
        runtime_where(runtime, datatype, n, w_data, w_stride, w_offset, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
        break;
    default:
        break;
    }
}

void runtime_reduction(reduction_operation_type_t reduction_operation_type, runtime_t runtime, datatype_t datatype, int64_t n,
                       void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        switch (reduction_operation_type)
        {
        case MAXIMUM_OPERATION:
            openblas_maximum(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        case SUMMATION_OPERATION:
            openblas_summation(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        default:
            break;
        }
        break;
    case MKL_RUNTIME:
        switch (reduction_operation_type)
        {
        case MAXIMUM_OPERATION:
            mkl_maximum(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        case SUMMATION_OPERATION:
            mkl_summation(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        default:
            break;
        }
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        switch (reduction_operation_type)
        {
        case MAXIMUM_OPERATION:
            cu_maximum(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        case SUMMATION_OPERATION:
            cu_summation(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
}

void runtime_image_to_column(datatype_t datatype, void *x_data, 
                             int64_t batch_size, int64_t channels, int64_t height, int64_t width, 
                             int64_t kernel_size, int64_t output_height, int64_t output_width,
                             int64_t stride, int64_t padding, void *y_data, bool_t inverse, void *padding_value)
{
    for (int64_t b = 0; b < batch_size; ++b)
    {
        int64_t b_offset_column = b * kernel_size * kernel_size * channels * output_height * output_width; 
        int64_t b_offset_image = b * channels * height * width;
        for (int64_t c = 0; c < channels * kernel_size * kernel_size; ++c) 
        {
            int64_t w_offset = c % kernel_size;
            int64_t h_offset = (c / kernel_size) % kernel_size;
            int64_t channel = c / kernel_size / kernel_size;
            for (int64_t h = 0; h < output_height; ++h)
            {
                int64_t row = h_offset + h * stride - padding;
                for (int64_t w = 0; w < output_width; ++w)
                {
                    int64_t column = w_offset + w * stride - padding;
                    int64_t column_index = b_offset_column + (c * output_height + h) * output_width + w;
                    int64_t image_index = column + width * (row + height * channel) + b_offset_image;
                    bool_t outside_boundary = row < 0 || column < 0 || row >= height || column >= width;
                    if (inverse)
                    {
                        if (!outside_boundary)
                        {
                            switch (datatype)
                            {
                            case FLOAT32:
                                ((float32_t *) y_data)[image_index] += ((float32_t *) x_data)[column_index];
                                break;
                            case FLOAT64:
                                ((float64_t *) y_data)[image_index] += ((float64_t *) x_data)[column_index];
                                break;
                            default:
                                break;
                            }
                        }
                    }
                    else
                    {
                        switch (datatype)
                        {
                        case FLOAT32:
                            ((float32_t *) y_data)[column_index] = (outside_boundary) ? *(float32_t *) padding_value : ((float32_t *) x_data)[image_index];
                            break;
                        case FLOAT64:
                            ((float64_t *) y_data)[column_index] = (outside_boundary) ? *(float64_t *) padding_value : ((float64_t *) x_data)[image_index];
                            break;
                        default:
                            break;
                        }
                    }
                }
            }
        }
    }
}

string_t runtime_string(runtime_t runtime)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        return "OPENBLAS_RUNTIME"; 
    case MKL_RUNTIME:
        return "MKL_RUNTIME";
    case CU_RUNTIME:
        return "CU_RUNTIME";
    default:
        return "RUNTIME";
    }
}

void runtime_zeroes(void *data, int64_t n, datatype_t datatype)
{
    for (int64_t i = 0; i < n; ++i)
    {
        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) 0.0;
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) 0.0;
            break;
        default:
            break;
        }
    }
}

void runtime_ones(void *data, int64_t n, datatype_t datatype)
{
    for (int64_t i = 0; i < n; ++i)
    {
        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) 1.0;
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) 1.0;
            break;
        default:
            break;
        }
    }
}

void runtime_arange(void *data, datatype_t datatype, void *start, void *stop, void *step)
{
    int64_t interval = 0;

    switch (datatype)
    {
    case FLOAT32:
        interval = (int64_t) ((*(float32_t *) stop - *(float32_t *) start) / *(float32_t *) step);
        break;
    case FLOAT64:
        interval = (int64_t) ((*(float64_t *) stop - *(float64_t *) start) / *(float64_t *) step);
        break;
    default:
        break;
    }

    for (int64_t i = 0; i < interval; ++i)
    {
        switch (datatype)
        {
        case FLOAT32:
            if (!i)
            {
                ((float32_t *) data)[i] = *(float32_t *) start;
            }
            else
            {
                ((float32_t *) data)[i] = ((float32_t *) data)[i - 1] + *(float32_t *) step;
            }
            break;
        case FLOAT64:
            if (!i)
            {
                ((float64_t *) data)[i] = *(float64_t *) start;
            }
            else
            {
                ((float64_t *) data)[i] = ((float64_t *) data)[i - 1] + *(float64_t *) step;
            }
            break;
        default:
            break;
        }
    }
}

void runtime_uniform(void *data, int64_t n, datatype_t datatype, void *lower_bound, void *upper_bound)
{
    for (int64_t i = 0; i < n; ++i)
    {
        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = uniformf(*(float32_t *) lower_bound, *(float32_t *) upper_bound);
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = uniform(*(float64_t *) lower_bound, *(float64_t *) upper_bound);
            break;
        default:
            break;
        }
    }
}

void runtime_normal(void *data, int64_t n, datatype_t datatype, void *mean, void *standard_deviation)
{
    for (int64_t i = 0; i < n; ++i)
    {
        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = normalf(*(float32_t *) mean, *(float32_t *) standard_deviation);
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = normal(*(float64_t *) mean, *(float64_t *) standard_deviation);
            break;
        default:
            break;
        }
    }
}
