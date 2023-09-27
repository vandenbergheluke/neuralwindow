#include <iostream>
extern "C"
{
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <graph.h>
}
#include <torch/torch.h>

#ifdef GRAPH

#define CASES 4

tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype);

nw_error_t *error;

tensor_t *tensors_x[CASES];
tensor_t *tensors_y[CASES];
tensor_t *tensors_x_output[CASES];
tensor_t *tensors_y_output[CASES];
tensor_t *returned_tensors[CASES];

torch::Tensor torch_tensors_x[CASES];
torch::Tensor torch_tensors_y[CASES];

std::vector<int64_t> shapes[CASES] = {
    {6, 5, 4, 3, 2},
    {6, 5, 4, 2, 1},
};

void setup(void)
{
    for (int i = 0; i < CASES; ++i)
    {
            tensors_x[i] = NULL;
            tensors_x_output[i] = NULL;
            tensors_y[i] = NULL;
            tensors_y_output[i] = NULL;
            returned_tensors[i] = NULL;

            torch_tensors_x[i] = torch::randn(shapes[0], torch::TensorOptions().dtype(torch::kFloat64));
            torch_tensors_y[i] = torch::randn(shapes[0], torch::TensorOptions().dtype(torch::kFloat64));

            tensors_x[i] = torch_to_tensor(torch_tensors_x[i], MKL_RUNTIME, FLOAT64);
            tensors_y[i] = torch_to_tensor(torch_tensors_y[i], MKL_RUNTIME, FLOAT64);
    }
}

void teardown(void)
{
    for (int i = 0; i < CASES; ++i)
    {

        tensor_destroy(tensors_x[i]);
        tensor_destroy(tensors_y[i]);
        tensor_destroy(tensors_x_output[i]);
        tensor_destroy(tensors_y_output[i]);
        tensor_destroy(returned_tensors[i]);
    }
    error_print(error);
    error_destroy(error);
    destroy_graph();
}

void test_graph(void)
{
    for (int i = 0; i < CASES; ++i)
    {
        error = tensor_rectified_linear(tensors_x[i], &tensors_x_output[i]);
        error = tensor_exponential(tensors_y[i], &tensors_y_output[i]);
        error = tensor_addition(tensors_x_output[i], tensors_y_output[i], &returned_tensors[i]);

        if (error)
        {
            error_print(error);
        }
    }
}


tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype)
{
    nw_error_t *error;
    view_t *view;
    storage_t *storage;
    buffer_t *buffer;
    tensor_t *tensor;

    switch (datatype)
    {
    case FLOAT32:
        torch_tensor = torch_tensor.to(torch::kFloat32);
        break;
    case FLOAT64:
        torch_tensor = torch_tensor.to(torch::kFloat64);
        break;
    default:
        return NULL;
    }

    error = view_create(&view, 
                        (uint64_t) torch_tensor.storage_offset(),
                        (uint64_t) torch_tensor.ndimension(),
                        (uint64_t *) torch_tensor.sizes().data(),
                        (uint64_t *) torch_tensor.strides().data());
    if (error)
    {
        error_print(error);
    }

    error = storage_create(&storage,
                           runtime,
                           datatype,
                           (uint64_t) torch_tensor.storage().nbytes() /
                           (uint64_t) datatype_size(datatype),
                           (void *) torch_tensor.data_ptr(),
                           true);
    if (error)
    {
        error_print(error);
    }

    error = buffer_create(&buffer, view, storage, false);
    if (error)
    {
        error_print(error);
    }

    error = tensor_create(&tensor, buffer, NULL, NULL, (bool_t) torch_tensor.requires_grad());
    if (error)
    {
        error_print(error);
    }

    return tensor;
}

#endif
int main(void)
{
    #ifdef GRAPH
        setup();
        test_graph();
        teardown(); 
    #endif  
}

