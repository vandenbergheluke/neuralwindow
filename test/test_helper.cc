#include <iostream>
extern "C"
{
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
}
#include <test_helper.h>
#include <torch/torch.h>

tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype)
{
    nw_error_t *error;
    view_t *view;
    storage_t *storage;
    buffer_t *buffer;
    tensor_t *tensor;

    error = view_create(&view, 
                        (uint64_t) torch_tensor.storage_offset(),
                        (uint64_t) torch_tensor.ndimension(),
                        (uint64_t *) torch_tensor.sizes().data(),
                        (uint64_t *) torch_tensor.strides().data());
    ck_assert_ptr_null(error);

    error = storage_create(&storage,
                           runtime,
                           datatype,
                           (uint64_t) torch_tensor.storage().nbytes() /
                           (uint64_t) datatype_size(datatype),
                           (void *) torch_tensor.data_ptr());
    ck_assert_ptr_null(error);

    error = buffer_create(&buffer,
                          view,
                          storage,
                          false);
    ck_assert_ptr_null(error);

    error = tensor_create(&tensor, buffer, NULL, NULL, true, true);
    ck_assert_ptr_null(error);

    return tensor;
}

void ck_assert_element_eq(const void *returned_data, uint64_t returned_index,
                          const void *expected_data, uint64_t expected_index,
                          datatype_t datatype)
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
                                   EPSILON);
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
                                    EPSILON);
        }
        break;
    default:
        ck_abort_msg("unknown datatype.");
    }
}

void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view)
{
    ck_assert_ptr_nonnull(returned_view);
    ck_assert_ptr_nonnull(expected_view);

    ck_assert_uint_eq(expected_view->rank, returned_view->rank);
    ck_assert_uint_eq(expected_view->offset, returned_view->offset);
    for (uint64_t i = 0; i < expected_view->rank; ++i)
    {
        ck_assert_uint_eq(expected_view->shape[i], returned_view->shape[i]);
        
        if (expected_view->shape[i] == 1)
        {
            ck_assert(returned_view->strides[i] == (uint64_t) 0 ||
                      expected_view->strides[i] == returned_view->strides[i]);
        }
        else
        {
            ck_assert_uint_eq(expected_view->strides[i], returned_view->strides[i]);
        }
    }
}

void ck_assert_storage_eq(const storage_t *returned_storage, const storage_t *expected_storage)
{
    ck_assert_ptr_nonnull(returned_storage);
    ck_assert_ptr_nonnull(expected_storage);
    
    ck_assert_uint_eq(expected_storage->n, returned_storage->n);
    ck_assert_int_eq(expected_storage->datatype, returned_storage->datatype);
    ck_assert_int_eq(expected_storage->runtime, returned_storage->runtime);
    for (uint64_t i = 0; i < expected_storage->n; ++i)
    {
        ck_assert_element_eq(returned_storage->data, i, 
                             expected_storage->data, i,
                             expected_storage->datatype);
    }
}

void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer)
{
    ck_assert_ptr_nonnull(returned_buffer);
    ck_assert_ptr_nonnull(expected_buffer);

    ck_assert_view_eq(returned_buffer->view, expected_buffer->view);
    ck_assert_storage_eq(returned_buffer->storage, expected_buffer->storage);
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
    PRINTLN_DEBUG_LOCATION("test");
    PRINTLN_DEBUG_TENSOR("returned", returned_tensor);
    PRINTLN_DEBUG_TENSOR("expected", expected_tensor);
    PRINT_DEBUG_NEWLINE;

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

void ck_assert_data_equiv(const void *returned_data, const uint64_t *returned_strides, uint64_t returned_offset,
                          const void *expected_data, const uint64_t *expected_strides, uint64_t expected_offset,
                          const uint64_t *shape, uint64_t rank, datatype_t datatype)
{
    switch (rank)
    {
    case 0:
        ck_assert_element_eq(returned_data, returned_offset, 
                             expected_data, expected_offset, 
                             datatype);
        break;
    case 1:
        for (uint64_t i = 0; i < shape[0]; ++i)
        {
            ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0], 
                                 expected_data, expected_offset + i * expected_strides[0],
                                 datatype);
        }
        break;
    case 2:
        for (uint64_t i = 0; i < shape[0]; ++i)
        {
            for (uint64_t j = 0; j < shape[1]; ++j)
            {
                ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1], 
                                     expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1],
                                     datatype);
            }
        }
        break;
    case 3:
        for (uint64_t i = 0; i < shape[0]; ++i)
        {
            for (uint64_t j = 0; j < shape[1]; ++j)
            {
                for (uint64_t k = 0; k < shape[2]; ++k)
                {
                    ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2], 
                                         expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2],
                                         datatype);
                }
            }
        }
        break;
    case 4:
        for (uint64_t i = 0; i < shape[0]; ++i)
        {
            for (uint64_t j = 0; j < shape[1]; ++j)
            {
                for (uint64_t k = 0; k < shape[2]; ++k)
                {
                    for (uint64_t l = 0; l < shape[3]; ++l)
                    {
                        ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2] + l * returned_strides[3], 
                                             expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2] + l * expected_strides[3],
                                             datatype);
                    }
                }
            }
        }
        break;
    case 5:
        for (uint64_t i = 0; i < shape[0]; ++i)
        {
            for (uint64_t j = 0; j < shape[1]; ++j)
            {
                for (uint64_t k = 0; k < shape[2]; ++k)
                {
                    for (uint64_t l = 0; l < shape[3]; ++l)
                    {
                        for (uint64_t m = 0; m < shape[4]; ++m)
                        {
                            ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2] + l * returned_strides[3] + m * returned_strides[4], 
                                                 expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2] + l * expected_strides[3] + m * expected_strides[4],
                                                 datatype);
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

void ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor)
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

    ck_assert_uint_eq(returned_tensor->buffer->view->rank, expected_tensor->buffer->view->rank);
    for (uint64_t i = 0; i < expected_tensor->buffer->view->rank; ++i)
    {
        ck_assert_uint_eq(returned_tensor->buffer->view->shape[i], 
                          expected_tensor->buffer->view->shape[i]);
    }
    ck_assert_int_eq(returned_tensor->buffer->storage->datatype, expected_tensor->buffer->storage->datatype);

    ck_assert_data_equiv(returned_tensor->buffer->storage->data, returned_tensor->buffer->view->strides, returned_tensor->buffer->view->offset,
                         expected_tensor->buffer->storage->data, expected_tensor->buffer->view->strides, expected_tensor->buffer->view->offset,
                         expected_tensor->buffer->view->shape, expected_tensor->buffer->view->rank, expected_tensor->buffer->storage->datatype);

}