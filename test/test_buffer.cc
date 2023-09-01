#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
}
#include <torch/torch.h>

#define EPSILON 0.0001
#define SEED 1234

bool_t set_seed = true;

#define UNARY_CASES 6

nw_error_t *unary_error;

buffer_t *unary_buffers[UNARY_CASES];
buffer_t *returned_unary_buffers[UNARY_CASES];
buffer_t *expected_unary_buffers[UNARY_CASES];

view_t *unary_views[UNARY_CASES];
view_t *returned_unary_views[UNARY_CASES];
view_t *expected_unary_views[UNARY_CASES];

torch::Tensor unary_tensors[UNARY_CASES];

void unary_setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    for (int i = 0; i < UNARY_CASES; ++i)
    {
        unary_buffers[i] = NULL;
        returned_unary_buffers[i] = NULL;
        expected_unary_buffers[i] = NULL;

        unary_views[i] = NULL;
        returned_unary_views[i] = NULL;
        expected_unary_views[i] = NULL;
    }

    std::vector<int64_t> shapes[UNARY_CASES] = {
        {1},
        {1},
        {1},
        {1},
        {3, 4, 5},
        {3, 4, 5},
    };
    
    runtime_t runtimes[UNARY_CASES] = {
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
    };

    datatype_t datatypes[UNARY_CASES] = {
        FLOAT32,
        FLOAT32,
        FLOAT64,
        FLOAT64,
        FLOAT32,
        FLOAT32,
    };

    torch::ScalarType torch_datatypes[UNARY_CASES] = {
        torch::kFloat32,
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat64,
        torch::kFloat32,
        torch::kFloat32,
    };

    for (int i = 0; i < UNARY_CASES; ++i)
    {
        unary_tensors[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));

        unary_error = view_create(&unary_views[i], 
                                  (uint64_t) unary_tensors[i].storage_offset(),
                                  (uint64_t) unary_tensors[i].ndimension(),
                                  (uint64_t *) unary_tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    unary_views[i],
                                    (void *) unary_tensors[i].data_ptr(),
                                    (uint64_t) unary_tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = view_create(&returned_unary_views[i],
                                  (uint64_t) unary_tensors[i].storage_offset(),
                                  (uint64_t) unary_tensors[i].ndimension(),
                                  (uint64_t *) unary_tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&returned_unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    returned_unary_views[i],
                                    NULL,
                                    (uint64_t) unary_tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_error);
    }
}

void unary_teardown(void)
{
    for (int i = 0; i < UNARY_CASES; i++)
    {
        buffer_destroy(unary_buffers[i]);
        buffer_destroy(returned_unary_buffers[i]);
        buffer_destroy(expected_unary_buffers[i]);
    }
    error_destroy(unary_error);
}

void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer)
{
    ck_assert_uint_eq(expected_buffer->view->rank, returned_buffer->view->rank);
    ck_assert_uint_eq(expected_buffer->view->offset, returned_buffer->view->offset);
    ck_assert_uint_eq(expected_buffer->n, returned_buffer->n);
    ck_assert_uint_eq(expected_buffer->size, returned_buffer->size);
    ck_assert_int_eq(expected_buffer->datatype, returned_buffer->datatype);
    ck_assert_int_eq(expected_buffer->runtime, returned_buffer->runtime);

    for (uint64_t i = 0; i < expected_buffer->view->rank; ++i)
    {
        ck_assert_uint_eq(expected_buffer->view->shape[i], returned_buffer->view->shape[i]);
        ck_assert_uint_eq(expected_buffer->view->strides[i], returned_buffer->view->strides[i]);
    }

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

START_TEST(test_exponential)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::exp(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_exponential(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_logarithm)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::log(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_logarithm(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_sine)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::sin(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_sine(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_cosine)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::cos(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_cosine(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_square_root)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::sqrt(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_square_root(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_reciprocal)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::reciprocal(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_reciprocal(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_copy)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::clone(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_copy(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_contiguous)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = unary_tensors[i].contiguous();

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_contiguous(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_negation)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::neg(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_negation(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_rectified_linear)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::relu(unary_tensors[i]);

        unary_error = view_create(&expected_unary_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_unary_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_rectified_linear(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

#define BINARY_CASES 4

nw_error_t *binary_error;

buffer_t *binary_buffers_x[BINARY_CASES];
buffer_t *binary_buffers_y[BINARY_CASES];
buffer_t *returned_binary_buffers[BINARY_CASES];
buffer_t *expected_binary_buffers[BINARY_CASES];

view_t *binary_views_x[BINARY_CASES];
view_t *binary_views_y[BINARY_CASES];
view_t *returned_binary_views[BINARY_CASES];
view_t *expected_binary_views[BINARY_CASES];

torch::Tensor binary_tensors_x[BINARY_CASES];
torch::Tensor binary_tensors_y[BINARY_CASES];

void binary_setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    for (int i = 0; i < BINARY_CASES; ++i)
    {
        binary_buffers_x[i] = NULL;
        binary_buffers_y[i] = NULL;
        returned_binary_buffers[i] = NULL;
        expected_binary_buffers[i] = NULL;

        binary_views_x[i] = NULL;
        binary_views_y[i] = NULL;
        returned_binary_views[i] = NULL;
        expected_binary_views[i] = NULL;
    }

    std::vector<int64_t> shapes[BINARY_CASES] = {
        {1},
        {1},
        {1},
        {1},
    };
    
    runtime_t runtimes[BINARY_CASES] = {
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
    };

    datatype_t datatypes[BINARY_CASES] = {
        FLOAT32,
        FLOAT32,
        FLOAT64,
        FLOAT64,
    };

    torch::ScalarType torch_datatypes[BINARY_CASES] = {
        torch::kFloat32,
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat64,
    };

    for (int i = 0; i < BINARY_CASES; ++i)
    {
        binary_tensors_x[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));
        binary_tensors_y[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));

        binary_error = view_create(&binary_views_x[i], 
                                   (uint64_t) binary_tensors_x[i].storage_offset(),
                                   (uint64_t) binary_tensors_x[i].ndimension(),
                                   (uint64_t *) binary_tensors_x[i].sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&binary_buffers_x[i],
                                     runtimes[i],
                                     datatypes[i],
                                     binary_views_x[i],
                                     (void *) binary_tensors_x[i].data_ptr(),
                                     (uint64_t) binary_tensors_x[i].numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = view_create(&binary_views_y[i], 
                                   (uint64_t) binary_tensors_y[i].storage_offset(),
                                   (uint64_t) binary_tensors_y[i].ndimension(),
                                   (uint64_t *) binary_tensors_y[i].sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&binary_buffers_y[i],
                                     runtimes[i],
                                     datatypes[i],
                                     binary_views_y[i],
                                     (void *) binary_tensors_y[i].data_ptr(),
                                     (uint64_t) binary_tensors_y[i].numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = view_create(&returned_binary_views[i],
                                   (uint64_t) binary_tensors_x[i].storage_offset(),
                                   (uint64_t) binary_tensors_x[i].ndimension(),
                                   (uint64_t *) binary_tensors_x[i].sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&returned_binary_buffers[i],
                                     runtimes[i],
                                     datatypes[i],
                                     returned_binary_views[i],
                                     NULL,
                                     (uint64_t) binary_tensors_x[i].numel(),
                                     true);
        ck_assert_ptr_null(binary_error);
    }
}

void binary_teardown(void)
{
    for (int i = 0; i < BINARY_CASES; i++)
    {
        buffer_destroy(binary_buffers_x[i]);
        buffer_destroy(binary_buffers_y[i]);
        buffer_destroy(returned_binary_buffers[i]);
        buffer_destroy(expected_binary_buffers[i]);
    }
    error_destroy(binary_error);
}

START_TEST(test_addition)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::add(binary_tensors_x[i], binary_tensors_y[i]);

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_addition(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

START_TEST(test_subtraction)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::subtract(binary_tensors_x[i], binary_tensors_y[i]);

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_subtraction(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

START_TEST(test_multiplication)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::mul(binary_tensors_x[i], binary_tensors_y[i]);

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_multiplication(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

START_TEST(test_division)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::div(binary_tensors_x[i], binary_tensors_y[i]);

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_division(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

START_TEST(test_power)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::pow(binary_tensors_x[i], binary_tensors_y[i]);

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_power(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

START_TEST(test_compare_equal)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::eq(binary_tensors_x[i], binary_tensors_y[i]).to(binary_tensors_x[i].dtype());

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_compare_equal(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

START_TEST(test_compare_greater)
{
    for (int i = 0; i < BINARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::gt(binary_tensors_x[i], binary_tensors_y[i]).to(binary_tensors_x[i].dtype());

        binary_error = view_create(&expected_binary_views[i],
                                   (uint64_t) expected_tensor.storage_offset(),
                                   (uint64_t) expected_tensor.ndimension(),
                                   (uint64_t *) expected_tensor.sizes().data(),
                                   NULL);
        ck_assert_ptr_null(binary_error);
        binary_error = buffer_create(&expected_binary_buffers[i],
                                     binary_buffers_x[i]->runtime,
                                     binary_buffers_x[i]->datatype,
                                     expected_binary_views[i],
                                     (void *) expected_tensor.data_ptr(),
                                     (uint64_t) expected_tensor.numel(),
                                     true);
        ck_assert_ptr_null(binary_error);

        binary_error = runtime_compare_greater(binary_buffers_x[i], binary_buffers_y[i], returned_binary_buffers[i]);
        ck_assert_ptr_null(binary_error);

        ck_assert_buffer_eq(returned_binary_buffers[i], expected_binary_buffers[i]);
    }
}
END_TEST

#define REDUCTION_CASES 4

nw_error_t *reduction_error;

buffer_t *reduction_buffers[REDUCTION_CASES];
buffer_t *returned_reduction_buffers[REDUCTION_CASES];
buffer_t *expected_reduction_buffers[REDUCTION_CASES];

view_t *reduction_views[REDUCTION_CASES];
view_t *returned_reduction_views[REDUCTION_CASES];
view_t *expected_reduction_views[REDUCTION_CASES];

torch::Tensor reduction_tensors[REDUCTION_CASES];

void reduction_setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    for (int i = 0; i < REDUCTION_CASES; ++i)
    {
        reduction_buffers[i] = NULL;
        returned_reduction_buffers[i] = NULL;
        expected_reduction_buffers[i] = NULL;

        reduction_views[i] = NULL;
        returned_reduction_views[i] = NULL;
        expected_reduction_views[i] = NULL;
    }

    std::vector<int64_t> shapes[REDUCTION_CASES] = {
        {2, 2},
        {2, 2},
        {2, 2},
        {2, 2},
    };
    
    runtime_t runtimes[REDUCTION_CASES] = {
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
    };

    datatype_t datatypes[REDUCTION_CASES] = {
        FLOAT32,
        FLOAT32,
        FLOAT64,
        FLOAT64,
    };

    torch::ScalarType torch_datatypes[REDUCTION_CASES] = {
        torch::kFloat32,
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat64,
    };

    for (int i = 0; i < REDUCTION_CASES; ++i)
    {
        reduction_tensors[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));

        reduction_error = view_create(&reduction_views[i], 
                                      (uint64_t) reduction_tensors[i].storage_offset(),
                                      (uint64_t) reduction_tensors[i].ndimension(),
                                      (uint64_t *) reduction_tensors[i].sizes().data(),
                                      NULL);
        ck_assert_ptr_null(reduction_error);
        unary_error = buffer_create(&reduction_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    reduction_views[i],
                                    (void *) reduction_tensors[i].data_ptr(),
                                    (uint64_t) reduction_tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(reduction_error);
    }
}

void reduction_teardown(void)
{
    for (int i = 0; i < REDUCTION_CASES; i++)
    {
        buffer_destroy(reduction_buffers[i]);
        buffer_destroy(returned_reduction_buffers[i]);
        buffer_destroy(expected_reduction_buffers[i]);
    }
    error_destroy(reduction_error);
}

START_TEST(test_summation)
{
    uint64_t axis[REDUCTION_CASES] = {
        0,
        0,
        1,
        1,
    };

    bool_t keep_dimension[REDUCTION_CASES] = {
        true,
        true,
        true,
        true,
    };

    for (int i = 0; i < REDUCTION_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::sum(reduction_tensors[i], std::vector<int64_t>({(int64_t) axis[i]}), keep_dimension[i]);

        reduction_error = view_create(&returned_reduction_views[i],
                                      (uint64_t) expected_tensor.storage_offset(),
                                      (uint64_t) expected_tensor.ndimension(),
                                      (uint64_t *) expected_tensor.sizes().data(),
                                      NULL);
        ck_assert_ptr_null(reduction_error);
        reduction_error = buffer_create(&returned_reduction_buffers[i],
                                        reduction_buffers[i]->runtime,
                                        reduction_buffers[i]->datatype,
                                        returned_reduction_views[i],
                                        NULL,
                                        (uint64_t) expected_tensor.numel(),
                                        true);
        ck_assert_ptr_null(reduction_error);
        reduction_error = view_create(&expected_reduction_views[i],
                                      (uint64_t) expected_tensor.storage_offset(),
                                      (uint64_t) expected_tensor.ndimension(),
                                      (uint64_t *) expected_tensor.sizes().data(),
                                      NULL);
        ck_assert_ptr_null(reduction_error);
        reduction_error = buffer_create(&expected_reduction_buffers[i],
                                        reduction_buffers[i]->runtime,
                                        reduction_buffers[i]->datatype,
                                        expected_reduction_views[i],
                                        (void *) expected_tensor.data_ptr(),
                                        (uint64_t) expected_tensor.numel(),
                                        true);
        ck_assert_ptr_null(reduction_error);

        reduction_error = runtime_summation(reduction_buffers[i], returned_reduction_buffers[i], axis[i]);
        ck_assert_ptr_null(reduction_error);

        ck_assert_buffer_eq(returned_reduction_buffers[i], expected_reduction_buffers[i]);
    }
}
END_TEST

START_TEST(test_maximum)
{
    uint64_t axis[REDUCTION_CASES] = {
        0,
        0,
        1,
        1,
    };

    bool_t keep_dimension[REDUCTION_CASES] = {
        true,
        true,
        true,
        true,
    };

    for (int i = 0; i < REDUCTION_CASES; ++i)
    {
        torch::Tensor expected_tensor = std::get<0>(torch::max(reduction_tensors[i], {(int64_t) axis[i]}, keep_dimension[i]));

        reduction_error = view_create(&returned_reduction_views[i],
                                      (uint64_t) expected_tensor.storage_offset(),
                                      (uint64_t) expected_tensor.ndimension(),
                                      (uint64_t *) expected_tensor.sizes().data(),
                                      NULL);
        ck_assert_ptr_null(reduction_error);
        reduction_error = buffer_create(&returned_reduction_buffers[i],
                                        reduction_buffers[i]->runtime,
                                        reduction_buffers[i]->datatype,
                                        returned_reduction_views[i],
                                        NULL,
                                        (uint64_t) expected_tensor.numel(),
                                        true);
        ck_assert_ptr_null(reduction_error);
        reduction_error = view_create(&expected_reduction_views[i],
                                      (uint64_t) expected_tensor.storage_offset(),
                                      (uint64_t) expected_tensor.ndimension(),
                                      (uint64_t *) expected_tensor.sizes().data(),
                                      NULL);
        ck_assert_ptr_null(reduction_error);
        reduction_error = buffer_create(&expected_reduction_buffers[i],
                                        reduction_buffers[i]->runtime,
                                        reduction_buffers[i]->datatype,
                                        expected_reduction_views[i],
                                        (void *) expected_tensor.data_ptr(),
                                        (uint64_t) expected_tensor.numel(),
                                        true);
        ck_assert_ptr_null(reduction_error);

        reduction_error = runtime_maximum(reduction_buffers[i], returned_reduction_buffers[i], axis[i]);
        ck_assert_ptr_null(reduction_error);

        ck_assert_buffer_eq(returned_reduction_buffers[i], expected_reduction_buffers[i]);
    }
}
END_TEST


Suite *make_buffer_suite(void)
{
    Suite *s;
    TCase *tc_unary;
    TCase *tc_binary;
    TCase *tc_reduction;

    s = suite_create("Test Buffer Suite");

    // Unary Operations
    tc_unary = tcase_create("Unary Case");
    tcase_add_checked_fixture(tc_unary, unary_setup, unary_teardown);
    tcase_add_test(tc_unary, test_exponential);
    tcase_add_test(tc_unary, test_logarithm);
    tcase_add_test(tc_unary, test_sine);
    tcase_add_test(tc_unary, test_cosine);
    tcase_add_test(tc_unary, test_square_root);
    tcase_add_test(tc_unary, test_reciprocal);
    tcase_add_test(tc_unary, test_copy);
    tcase_add_test(tc_unary, test_contiguous);
    tcase_add_test(tc_unary, test_negation);
    tcase_add_test(tc_unary, test_rectified_linear);

    // Binary Operations
    tc_binary = tcase_create("Binary Case");
    tcase_add_checked_fixture(tc_binary, binary_setup, binary_teardown);
    tcase_add_test(tc_binary, test_addition);
    tcase_add_test(tc_binary, test_subtraction);
    tcase_add_test(tc_binary, test_multiplication);
    tcase_add_test(tc_binary, test_division);
    tcase_add_test(tc_binary, test_power);
    tcase_add_test(tc_binary, test_compare_equal);
    tcase_add_test(tc_binary, test_compare_greater);

    // Reduction Operations
    tc_reduction = tcase_create("Reduction Case");
    tcase_add_checked_fixture(tc_reduction, reduction_setup, reduction_teardown);
    tcase_add_test(tc_reduction, test_summation);
    tcase_add_test(tc_reduction, test_maximum);

    suite_add_tcase(s, tc_unary);
    suite_add_tcase(s, tc_binary);
    suite_add_tcase(s, tc_reduction);

    return s;
}

extern "C" int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
