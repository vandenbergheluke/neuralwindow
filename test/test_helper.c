#include <view.h>
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <runtime.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <test_helper.h>

static inline float32_t get_epsilon_float(float32_t a, float32_t b, float32_t abs_epsilon_f)
{
    static float32_t epsilon_f = 128 * FLT_EPSILON;
    return MAX(abs_epsilon_f, epsilon_f * MIN((ABS(a) + ABS(b)), FLT_MAX));
}

static inline float64_t get_epsilon_double(float64_t a, float64_t b, float64_t abs_epsilon)
{
    static float64_t epsilon = 1e2 * FLT_EPSILON;
    return MAX(abs_epsilon, epsilon * MIN((ABS(a) + ABS(b)), FLT_MAX));
}

void ck_assert_element_eq(const void *returned_data, int64_t returned_index,
                          const void *expected_data, int64_t expected_index,
                          datatype_t datatype, void *epsilon)
{
    ck_assert_ptr_nonnull(returned_data);
    ck_assert_ptr_nonnull(expected_data);

    switch (datatype)
    {
    case FLOAT32:
        if (isnanf(((float32_t *) expected_data)[expected_index]))
        {
            ck_assert_float_nan(((float32_t *) returned_data)[returned_index]);
        }
        else
        {
            ck_assert_float_eq_tol(((float32_t *) returned_data)[returned_index],
                                   ((float32_t *) expected_data)[expected_index],
                                   (!epsilon) ? get_epsilon_float(((float32_t *) returned_data)[returned_index],
                                                                  ((float32_t *) expected_data)[expected_index], 1e-5) : 
                                                get_epsilon_float(((float32_t *) returned_data)[returned_index],
                                                                  ((float32_t *) expected_data)[expected_index], *(float32_t *) epsilon));
        }
        break;
    case FLOAT64:
        if (isnanf(((float64_t *) expected_data)[expected_index]))
        {
            ck_assert_double_nan(((float64_t *) returned_data)[returned_index]);
        }
        else
        {
            ck_assert_double_eq_tol(((float64_t *) returned_data)[returned_index],
                                    ((float64_t *) expected_data)[expected_index],
                                    (!epsilon) ? get_epsilon_double(((float64_t *) returned_data)[returned_index],
                                                                    ((float64_t *) expected_data)[expected_index], 1e-5) : 
                                                 get_epsilon_double(((float64_t *) returned_data)[returned_index],
                                                                    ((float64_t *) expected_data)[expected_index], *(float64_t *) epsilon));
        }
        break;
    default:
        ck_abort_msg("unknown datatype.");
    }
}

void ck_assert_storage_eq(const storage_t *returned_storage, const storage_t *expected_storage, void *epsilon)
{
    ck_assert_ptr_nonnull(returned_storage);
    ck_assert_ptr_nonnull(expected_storage);
    
    ck_assert_int_eq(expected_storage->n, returned_storage->n);
    ck_assert_int_eq(expected_storage->datatype, returned_storage->datatype);
    ck_assert_int_eq(expected_storage->runtime, returned_storage->runtime);
    for (int64_t i = 0; i < expected_storage->n; ++i)
    {
        ck_assert_element_eq(returned_storage->data, i, expected_storage->data, i, expected_storage->datatype, epsilon);
    }
}

void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer, void *epsilon)
{
    ck_assert_ptr_nonnull(returned_buffer);
    ck_assert_ptr_nonnull(expected_buffer);

    ck_assert_view_eq(returned_buffer->view, expected_buffer->view);
    ck_assert_storage_eq(returned_buffer->storage, expected_buffer->storage, epsilon);
}

void ck_assert_function_eq(const tensor_t *returned_tensor, 
                           const function_t *returned_function,
                           const tensor_t *expected_tensor,
                           const function_t *expected_function)
{
    ck_assert_ptr_nonnull(returned_tensor);
    ck_assert_ptr_nonnull(expected_tensor);
    ck_assert_ptr_nonnull(returned_function);
    ck_assert_ptr_nonnull(expected_function);

    ck_assert_int_eq(expected_function->operation_type,
                     returned_function->operation_type);
    switch (expected_function->operation_type)
    {
    case UNARY_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->unary_operation->x,
                            returned_function->operation->unary_operation->x);
        ck_assert_int_eq(expected_function->operation->unary_operation->operation_type,
                         returned_function->operation->unary_operation->operation_type);
        break;
    case BINARY_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->binary_operation->x,
                            returned_function->operation->binary_operation->x);
        ck_assert_tensor_eq(expected_function->operation->binary_operation->y,
                            returned_function->operation->binary_operation->y);
        ck_assert_int_eq(expected_function->operation->binary_operation->operation_type,
                         returned_function->operation->binary_operation->operation_type);
        break;
    case REDUCTION_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->reduction_operation->x,
                            returned_function->operation->reduction_operation->x);
        ck_assert_int_eq(expected_function->operation->reduction_operation->operation_type,
                         returned_function->operation->reduction_operation->operation_type);
        ck_assert_int_eq(expected_function->operation->reduction_operation->length,
                          returned_function->operation->reduction_operation->length);
        ck_assert(expected_function->operation->reduction_operation->keep_dimension == 
                  returned_function->operation->reduction_operation->keep_dimension);
        for (int64_t j = 0; j < expected_function->operation->reduction_operation->length; ++j)
        {
            ck_assert_int_eq(expected_function->operation->reduction_operation->axis[j],
                              returned_function->operation->reduction_operation->axis[j]);
        }
        break;
    case STRUCTURE_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->structure_operation->x,
                            returned_function->operation->structure_operation->x);
        ck_assert_int_eq(expected_function->operation->structure_operation->operation_type,
                         returned_function->operation->structure_operation->operation_type);
        ck_assert_int_eq(expected_function->operation->structure_operation->length,
                          returned_function->operation->structure_operation->length);
        for (int64_t j = 0; j < expected_function->operation->structure_operation->length; ++j)
        {
            ck_assert_int_eq(expected_function->operation->structure_operation->arguments[j],
                              returned_function->operation->structure_operation->arguments[j]);
        }
        break;
    default:
        ck_abort_msg("unknown operation type");
    } 
}

void ck_assert_tensor_eq(const tensor_t *returned_tensor, const tensor_t *expected_tensor)
{
    PRINTLN_DEBUG_LOCATION("test");
    PRINTLN_DEBUG_TENSOR("returned", returned_tensor);
    PRINTLN_DEBUG_TENSOR("expected", expected_tensor);
    PRINT_DEBUG_NEWLINE;

    if (!expected_tensor)
    {
        ck_assert_ptr_null(returned_tensor);
        return;
    }

    if (!expected_tensor->buffer)
    {
        ck_assert_ptr_null(expected_tensor->buffer);
    }
    else
    {
        ck_assert_buffer_eq(returned_tensor->buffer, expected_tensor->buffer, NULL);
    }

    ck_assert(returned_tensor->requires_gradient == expected_tensor->requires_gradient);
}

void ck_assert_data_equiv(const void *returned_data, const int64_t *returned_strides, int64_t returned_offset,
                          const void *expected_data, const int64_t *expected_strides, int64_t expected_offset,
                          const int64_t *shape, int64_t rank, datatype_t datatype, void *epsilon)
{
    switch (rank)
    {
    case 0:
        ck_assert_element_eq(returned_data, returned_offset, 
                             expected_data, expected_offset, 
                             datatype, epsilon);
        break;
    case 1:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0], 
                                 expected_data, expected_offset + i * expected_strides[0],
                                 datatype, epsilon);
        }
        break;
    case 2:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1], 
                                     expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1],
                                     datatype, epsilon);
            }
        }
        break;
    case 3:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                for (int64_t k = 0; k < shape[2]; ++k)
                {
                    ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2], 
                                         expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2],
                                         datatype, epsilon);
                }
            }
        }
        break;
    case 4:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                for (int64_t k = 0; k < shape[2]; ++k)
                {
                    for (int64_t l = 0; l < shape[3]; ++l)
                    {
                        ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2] + l * returned_strides[3], 
                                             expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2] + l * expected_strides[3],
                                             datatype, epsilon);
                    }
                }
            }
        }
        break;
    case 5:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                for (int64_t k = 0; k < shape[2]; ++k)
                {
                    for (int64_t l = 0; l < shape[3]; ++l)
                    {
                        for (int64_t m = 0; m < shape[4]; ++m)
                        {
                            ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2] + l * returned_strides[3] + m * returned_strides[4], 
                                                 expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2] + l * expected_strides[3] + m * expected_strides[4],
                                                 datatype, epsilon);
                        }
                    }
                }
            }
        }
        break;
    default:
        ck_abort_msg("unsupported rank.");
    }
}

static void _ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor, void *epsilon)
{
    PRINTLN_DEBUG_LOCATION("test");
    PRINTLN_DEBUG_TENSOR("returned", returned_tensor);
    PRINTLN_DEBUG_TENSOR("expected", expected_tensor);
    PRINT_DEBUG_NEWLINE;

    ck_assert_ptr_nonnull(expected_tensor->buffer);
    ck_assert_ptr_nonnull(expected_tensor->buffer->view);
    ck_assert_ptr_nonnull(expected_tensor->buffer->storage);
    ck_assert_ptr_nonnull(returned_tensor->buffer);
    ck_assert_ptr_nonnull(returned_tensor->buffer->view);
    ck_assert_ptr_nonnull(returned_tensor->buffer->storage);

    ck_assert_int_eq(returned_tensor->buffer->view->rank, expected_tensor->buffer->view->rank);
    for (int64_t i = 0; i < expected_tensor->buffer->view->rank; ++i)
    {
        ck_assert_int_eq(returned_tensor->buffer->view->shape[i], 
                          expected_tensor->buffer->view->shape[i]);
    }
    ck_assert_int_eq(returned_tensor->buffer->storage->datatype, expected_tensor->buffer->storage->datatype);

    ck_assert_data_equiv(returned_tensor->buffer->storage->data, returned_tensor->buffer->view->strides, returned_tensor->buffer->view->offset,
                         expected_tensor->buffer->storage->data, expected_tensor->buffer->view->strides, expected_tensor->buffer->view->offset,
                         expected_tensor->buffer->view->shape, expected_tensor->buffer->view->rank, expected_tensor->buffer->storage->datatype, epsilon);

}

void ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor)
{
    _ck_assert_tensor_equiv(returned_tensor, expected_tensor, NULL);
}

void ck_assert_tensor_equiv_flt(const tensor_t *returned_tensor, const tensor_t *expected_tensor, float32_t abs_epsilon)
{
    _ck_assert_tensor_equiv(returned_tensor, expected_tensor, &abs_epsilon);
}

void ck_assert_tensor_equiv_dbl(const tensor_t *returned_tensor, const tensor_t *expected_tensor, float64_t abs_epsilon)
{
    _ck_assert_tensor_equiv(returned_tensor, expected_tensor, &abs_epsilon);
}

void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view)
{
    ck_assert_ptr_nonnull(returned_view);
    ck_assert_ptr_nonnull(expected_view);

    ck_assert_int_eq(expected_view->rank, returned_view->rank);
    ck_assert_int_eq(expected_view->offset, returned_view->offset);
    for (int64_t i = 0; i < expected_view->rank; ++i)
    {
        ck_assert_int_eq(expected_view->shape[i], returned_view->shape[i]);
        
        if (expected_view->shape[i] == 1)
        {
            ck_assert(returned_view->strides[i] == (int64_t) 0 || expected_view->strides[i] == returned_view->strides[i]);
        }
        else
        {
            ck_assert_int_eq(expected_view->strides[i], returned_view->strides[i]);
        }
    }
}
