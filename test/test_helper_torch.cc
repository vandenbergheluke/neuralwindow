#include <iostream>
extern "C"
{
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <runtime.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
}
#include <test_helper_torch.h>
#include <torch/torch.h>

tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype)
{
    nw_error_t *error = NULL;
    buffer_t *buffer = NULL;
    storage_t *storage = NULL;
    view_t *view = NULL;
    tensor_t *tensor = NULL;

    switch (datatype)
    {
    case FLOAT32:
        torch_tensor = torch_tensor.to(torch::kFloat32);
        break;
    case FLOAT64:
        torch_tensor = torch_tensor.to(torch::kFloat64);
        break;
    default:
        ck_abort_msg("invalid datatype.");
    }

    error = storage_create(&storage, runtime, datatype, torch_tensor.storage().nbytes() / datatype_size(datatype), (void *) torch_tensor.storage().data_ptr().get(), true);
    if (error)
    {
        error_print(error);
    }
    ck_assert_ptr_null(error);
    
    error = view_create(&view, torch_tensor.storage_offset(), torch_tensor.ndimension(), torch_tensor.sizes().data(), torch_tensor.strides().data());
    if (error)
    {
        error_print(error);
    }
    ck_assert_ptr_null(error);

    error = buffer_create(&buffer, view, storage, false);
    if (error)
    {
        error_print(error);
    }
    ck_assert_ptr_null(error);

    error = tensor_create(&tensor, buffer, NULL, NULL, (bool_t) torch_tensor.requires_grad(), true);
    if (error)
    {
        error_print(error);
    }
    ck_assert_ptr_null(error);

    return tensor;
}
