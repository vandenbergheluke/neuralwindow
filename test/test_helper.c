#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <test_helper.h>

void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view)
{
    ck_assert_uint_eq(expected_view->rank, returned_view->rank);
    ck_assert_uint_eq(expected_view->offset, returned_view->offset);
    for (uint64_t i = 0; i < expected_view->rank; ++i)
    {
        ck_assert_uint_eq(expected_view->shape[i], returned_view->shape[i]);
        ck_assert_uint_eq(expected_view->strides[i], returned_view->strides[i]);
    }
}

void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer)
{
    ck_assert_uint_eq(expected_buffer->n, returned_buffer->n);
    ck_assert_uint_eq(expected_buffer->size, returned_buffer->size);
    ck_assert_int_eq(expected_buffer->datatype, returned_buffer->datatype);
    ck_assert_int_eq(expected_buffer->runtime, returned_buffer->runtime);
    ck_assert_view_eq(returned_buffer->view, expected_buffer->view);

    for (uint64_t i = 0; i < expected_buffer->n; ++i)
    {

        switch (expected_buffer->datatype)
        {
        case FLOAT32:
            if (isnanf(((float32_t *) expected_buffer->data)[i]))
            {
                ck_assert_float_nan(((float32_t *) returned_buffer->data)[i]);
            }
            else
            {
                ck_assert_float_eq_tol(((float32_t *) returned_buffer->data)[i],
                                       ((float32_t *) expected_buffer->data)[i], EPSILON);
            }
            break;
        case FLOAT64:
            if (isnan(((float64_t *) expected_buffer->data)[i]))
            {
                ck_assert_double_nan(((float64_t *) returned_buffer->data)[i]);
            }
            else
            {
                ck_assert_double_eq_tol(((float64_t *) returned_buffer->data)[i],
                                        ((float64_t *) expected_buffer->data)[i], EPSILON);
            }
        default:
            break;
        }
    }
}

void ck_assert_function_eq(const tensor_t *returned_tensor, 
                           const function_t *returned_function,
                           const tensor_t *expected_tensor,
                           const function_t *expected_function)
{
    ck_assert_int_eq(expected_function->operation_type,
                        returned_function->operation_type);
    switch (expected_function->operation_type)
    {
    case UNARY_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->unary_operation->x,
                            returned_function->operation->unary_operation->x);
        ck_assert_ptr_eq(returned_tensor,
                         returned_function->operation->unary_operation->result);
        ck_assert_ptr_eq(expected_tensor,
                         expected_function->operation->unary_operation->result);
        ck_assert_int_eq(expected_function->operation->unary_operation->operation_type,
                         returned_function->operation->unary_operation->operation_type);
        break;
    case BINARY_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->binary_operation->x,
                            returned_function->operation->binary_operation->x);
        ck_assert_tensor_eq(expected_function->operation->binary_operation->y,
                            returned_function->operation->binary_operation->y);
        ck_assert_ptr_eq(returned_tensor,
                         returned_function->operation->binary_operation->result);
        ck_assert_ptr_eq(expected_tensor,
                         expected_function->operation->binary_operation->result);
        ck_assert_int_eq(expected_function->operation->binary_operation->operation_type,
                         returned_function->operation->binary_operation->operation_type);
        break;
    case REDUCTION_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->reduction_operation->x,
                            returned_function->operation->reduction_operation->x);
        ck_assert_ptr_eq(returned_tensor,
                         returned_function->operation->reduction_operation->result);
        ck_assert_ptr_eq(expected_tensor,
                         expected_function->operation->reduction_operation->result);
        ck_assert_int_eq(expected_function->operation->reduction_operation->operation_type,
                         returned_function->operation->reduction_operation->operation_type);
        ck_assert_uint_eq(expected_function->operation->reduction_operation->length,
                          returned_function->operation->reduction_operation->length);
        ck_assert(expected_function->operation->reduction_operation->keep_dimension == 
                  returned_function->operation->reduction_operation->keep_dimension);
        for (uint64_t j = 0; j < expected_function->operation->reduction_operation->length; ++j)
        {
            ck_assert_uint_eq(expected_function->operation->reduction_operation->axis[j],
                              returned_function->operation->reduction_operation->axis[j]);
        }
        break;
    case STRUCTURE_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->structure_operation->x,
                            returned_function->operation->structure_operation->x);
        ck_assert_ptr_eq(returned_tensor,
                         returned_function->operation->structure_operation->result);
        ck_assert_ptr_eq(expected_tensor,
                         expected_function->operation->structure_operation->result);
        ck_assert_int_eq(expected_function->operation->structure_operation->operation_type,
                         returned_function->operation->structure_operation->operation_type);
        ck_assert_uint_eq(expected_function->operation->structure_operation->length,
                          returned_function->operation->structure_operation->length);
        for (uint64_t j = 0; j < expected_function->operation->structure_operation->length; ++j)
        {
            ck_assert_uint_eq(expected_function->operation->structure_operation->arguments[j],
                              returned_function->operation->structure_operation->arguments[j]);
        }
        break;
    default:
        ck_abort_msg("unknown operation type");
    } 
}

void ck_assert_tensor_eq(const tensor_t *returned_tensor, const tensor_t *expected_tensor)
{
    if (expected_tensor == NULL)
    {
        ck_assert_ptr_null(returned_tensor);
        return;
    }

    if (expected_tensor->buffer == NULL)
    {
        ck_assert_ptr_null(expected_tensor->buffer);
    }
    else
    {
        ck_assert_buffer_eq(returned_tensor->buffer, expected_tensor->buffer);
    }

    if (expected_tensor->context == NULL)
    {
        ck_assert_ptr_null(returned_tensor->context);
    }
    else
    {
        ck_assert_function_eq(returned_tensor, returned_tensor->context,
                              expected_tensor, expected_tensor->context);
    }

    ck_assert(returned_tensor->requires_gradient == expected_tensor->requires_gradient);
}