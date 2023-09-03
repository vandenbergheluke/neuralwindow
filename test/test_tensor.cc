#include "datatype.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/sum.h>
#include <iostream>
extern "C"
{
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
}
#include <torch/torch.h>
#include <cstring>

#define EPSILON 0.00001

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
        ck_assert_uint_eq(expected_tensor->buffer->view->rank, returned_tensor->buffer->view->rank);
        ck_assert_uint_eq(expected_tensor->buffer->view->offset, returned_tensor->buffer->view->offset);
        ck_assert_uint_eq(expected_tensor->buffer->n, returned_tensor->buffer->n);
        ck_assert_uint_eq(expected_tensor->buffer->size, returned_tensor->buffer->size);
        ck_assert_int_eq(expected_tensor->buffer->datatype, returned_tensor->buffer->datatype);
        ck_assert_int_eq(expected_tensor->buffer->runtime, returned_tensor->buffer->runtime);

        for (uint64_t i = 0; i < expected_tensor->buffer->view->rank; ++i)
        {
            ck_assert_uint_eq(expected_tensor->buffer->view->shape[i], returned_tensor->buffer->view->shape[i]);
            ck_assert_uint_eq(expected_tensor->buffer->view->strides[i], returned_tensor->buffer->view->strides[i]);
        }

        for (uint64_t i = 0; i < expected_tensor->buffer->n; ++i)
        {

            switch (expected_tensor->buffer->datatype)
            {
            case FLOAT32:
                if (isnanf(((float32_t *) expected_tensor->buffer->data)[i]))
                {
                    ck_assert_float_nan(((float32_t *) returned_tensor->buffer->data)[i]);
                }
                else
                {
                    ck_assert_float_eq_tol(((float32_t *) returned_tensor->buffer->data)[i],
                                        ((float32_t *) expected_tensor->buffer->data)[i], EPSILON);
                }
                break;
            case FLOAT64:
                if (isnan(((float64_t *) expected_tensor->buffer->data)[i]))
                {
                    ck_assert_double_nan(((float64_t *) returned_tensor->buffer->data)[i]);
                }
                else
                {
                    ck_assert_double_eq_tol(((float64_t *) returned_tensor->buffer->data)[i],
                                            ((float64_t *) expected_tensor->buffer->data)[i], EPSILON);
                }
            default:
                break;
            }
        }
    }

    if (expected_tensor->context == NULL)
    {
        ck_assert_ptr_null(returned_tensor->context);
    }
    else
    {
        ck_assert_int_eq(expected_tensor->context->operation_type,
                         returned_tensor->context->operation_type);
        switch (expected_tensor->context->operation_type)
        {
        case UNARY_OPERATION:
            ck_assert_tensor_eq(expected_tensor->context->operation->unary_operation->x,
                                returned_tensor->context->operation->unary_operation->x);
            ck_assert_ptr_eq(returned_tensor,
                             returned_tensor->context->operation->unary_operation->result);
            ck_assert_int_eq(expected_tensor->context->operation->unary_operation->operation_type,
                             returned_tensor->context->operation->unary_operation->operation_type);
            break;
        case BINARY_OPERATION:
            ck_assert_tensor_eq(expected_tensor->context->operation->binary_operation->x,
                                returned_tensor->context->operation->binary_operation->x);
            ck_assert_tensor_eq(expected_tensor->context->operation->binary_operation->y,
                                returned_tensor->context->operation->binary_operation->y);
            ck_assert_ptr_eq(returned_tensor,
                             returned_tensor->context->operation->binary_operation->result);
            ck_assert_int_eq(expected_tensor->context->operation->binary_operation->operation_type,
                             returned_tensor->context->operation->binary_operation->operation_type);
            break;
        case REDUCTION_OPERATION:
            ck_assert_tensor_eq(expected_tensor->context->operation->reduction_operation->x,
                                returned_tensor->context->operation->reduction_operation->x);
            ck_assert_ptr_eq(returned_tensor,
                             returned_tensor->context->operation->reduction_operation->result);
            ck_assert_int_eq(expected_tensor->context->operation->reduction_operation->operation_type,
                             returned_tensor->context->operation->reduction_operation->operation_type);
            ck_assert_uint_eq(expected_tensor->context->operation->reduction_operation->length,
                              returned_tensor->context->operation->reduction_operation->length);
            ck_assert(expected_tensor->context->operation->reduction_operation->keep_dimension == 
                      returned_tensor->context->operation->reduction_operation->keep_dimension);
            for (uint64_t j = 0; j < expected_tensor->context->operation->reduction_operation->length; ++j)
            {
                ck_assert_uint_eq(expected_tensor->context->operation->reduction_operation->axis[j],
                                  returned_tensor->context->operation->reduction_operation->axis[j]);
            }
            break;
        case STRUCTURE_OPERATION:
            ck_assert_tensor_eq(expected_tensor->context->operation->structure_operation->x,
                                returned_tensor->context->operation->structure_operation->x);
            ck_assert_ptr_eq(returned_tensor,
                             returned_tensor->context->operation->structure_operation->result);
            ck_assert_int_eq(expected_tensor->context->operation->structure_operation->operation_type,
                             returned_tensor->context->operation->structure_operation->operation_type);
            ck_assert_uint_eq(expected_tensor->context->operation->structure_operation->length,
                              returned_tensor->context->operation->structure_operation->length);
            for (uint64_t j = 0; j < expected_tensor->context->operation->structure_operation->length; ++j)
            {
                ck_assert_uint_eq(expected_tensor->context->operation->structure_operation->arguments[j],
                                  returned_tensor->context->operation->structure_operation->arguments[j]);
            }
            break;
        default:
            ck_abort_msg("unknown operation type");
        } 
    }

    ck_assert(returned_tensor->requires_gradient == expected_tensor->requires_gradient);
}

#define SEED 1234

bool_t set_seed = true;

#define UNARY_CASES 6

nw_error_t *unary_error;

buffer_t *unary_buffers[UNARY_CASES];
buffer_t *expected_unary_buffers[UNARY_CASES];

view_t *unary_views[UNARY_CASES];
view_t *expected_unary_views[UNARY_CASES];

tensor_t *unary_tensors[UNARY_CASES];
tensor_t *returned_unary_tensors[UNARY_CASES];
tensor_t *expected_unary_tensors[UNARY_CASES];

torch::Tensor unary_torch_tensors[UNARY_CASES];

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
        expected_unary_buffers[i] = NULL;

        unary_views[i] = NULL;
        expected_unary_views[i] = NULL;

        unary_tensors[i] = NULL;
        returned_unary_tensors[i] = NULL;
        expected_unary_tensors[i] = NULL;
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
        unary_torch_tensors[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));

        unary_error = view_create(&unary_views[i], 
                                  (uint64_t) unary_torch_tensors[i].storage_offset(),
                                  (uint64_t) unary_torch_tensors[i].ndimension(),
                                  (uint64_t *) unary_torch_tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);

        unary_error = buffer_create(&unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    unary_views[i],
                                    (void *) unary_torch_tensors[i].data_ptr(),
                                    (uint64_t) unary_torch_tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = tensor_create(&unary_tensors[i],
                                    unary_buffers[i],
                                    NULL,
                                    NULL,
                                    false,
                                    false);
        ck_assert_ptr_null(unary_error);

        unary_error = tensor_create_empty(&expected_unary_tensors[i]);
        ck_assert_ptr_null(unary_error);

        unary_error = tensor_create_empty(&returned_unary_tensors[i]);
        ck_assert_ptr_null(unary_error);
    }
}

void unary_teardown(void)
{
    for (int i = 0; i < UNARY_CASES; i++)
    {
        tensor_destroy(unary_tensors[i]);
        tensor_destroy(expected_unary_tensors[i]);
        tensor_destroy(returned_unary_tensors[i]);
    }
    if (unary_error != NULL)
    {
        error_print(unary_error);
    }
    error_destroy(unary_error);
}

START_TEST(test_exponential_forward)
{
    for (int i = 0; i < UNARY_CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::exp(unary_torch_tensors[i]);

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

        function_t *function = NULL; 
        operation_t *operation = NULL;
        unary_operation_t *unary_operation = NULL;

        unary_error = unary_operation_create(&unary_operation,
                                             EXPONENTIAL_OPERATION,
                                             unary_tensors[i],
                                             expected_unary_tensors[i]);
        ck_assert_ptr_null(unary_error);
        unary_error = operation_create(&operation, UNARY_OPERATION, unary_operation);
        ck_assert_ptr_null(unary_error);
        unary_error = function_create(&function, operation, UNARY_OPERATION);
        ck_assert_ptr_null(unary_error);

        expected_unary_tensors[i]->buffer = expected_unary_buffers[i];
        expected_unary_tensors[i]->context = function;
        expected_unary_tensors[i]->requires_gradient = unary_tensors[i]->requires_gradient;
        expected_unary_tensors[i]->lock = unary_tensors[i]->lock;

        unary_error = tensor_exponential(unary_tensors[i], returned_unary_tensors[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_tensor_eq(returned_unary_tensors[i], expected_unary_tensors[i]);
    }
}
END_TEST

// nw_error_t *error;

// runtime_t runtimes[] = {
//    OPENBLAS_RUNTIME,
//    MKL_RUNTIME,
// //    CU_RUNTIME
// };

// void setup(void)
// {
//     if (set_seed)
//     {
//         torch::manual_seed(SEED);
//         set_seed = false;
//     }

//     error = NULL;
// }

// void teardown(void)
// {
//     error_print(error);
//     error_destroy(error);
// }


// START_TEST(test_exponential_backward)
// {
//     torch::Tensor X = torch::randn({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
//     torch::Tensor Y = torch::exp(X);
//     torch::Tensor J = torch::sum(Y);
//     J.backward();

//     view_t *view;
//     buffer_t *buffer;
//     tensor_t *nw_X;
//     tensor_t *nw_Y;
//     tensor_t *nw_J;

//     for (long unsigned int i = 0; i < (sizeof(runtimes) / sizeof(runtimes[0])); i++) {
//         error = view_create(&view, 
//                             (uint64_t) X.storage_offset(),
//                             (uint64_t) X.ndimension(),
//                             (uint64_t *) X.sizes().data(),
//                             NULL);
//         ck_assert_ptr_null(error);
//         error = buffer_create(&buffer,
//                               runtimes[i],
//                               FLOAT32,
//                               view,
//                               (void *) X.data_ptr(),
//                               (uint64_t) X.numel(),
//                               true);
//         ck_assert_ptr_null(error);
//         error = tensor_create(&nw_X,
//                               buffer,
//                               NULL,
//                               NULL,
//                               true,
//                               true);
//         ck_assert_ptr_null(error);

//         error = tensor_create_empty(&nw_Y);
//         ck_assert_ptr_null(error);

//         error = tensor_create_empty(&nw_J);
//         ck_assert_ptr_null(error);

//         error = tensor_exponential(nw_X, nw_Y);
//         ck_assert_ptr_null(error);

//         error = tensor_summation(nw_Y,
//                                  nw_J,
//                                  NULL,
//                                  (uint64_t) X.ndimension(),
//                                  false);
//         ck_assert_ptr_null(error);

//         error = tensor_backward(nw_J, NULL);
//         ck_assert_ptr_null(error);

//         // printf("%lu\n", nw_X->gradient->buffer->view->shape[0]);
//         // printf("%lu\n", nw_X->gradient->buffer->view->shape[1]);
//         // printf("%lu\n", nw_X->gradient->buffer->view->shape[0]);
//         for (uint64_t j = 0; j < nw_X->gradient->buffer->n; ++j)
//             printf("%f, ", ((float *) nw_X->gradient->buffer->data)[j]);
//         printf("\n");
//         for (uint64_t j = 0; j < nw_X->gradient->buffer->n; ++j)
//             printf("%f, ", ((float *) X.grad().data_ptr())[j]);
//         printf("\n");
//         ck_assert(std::memcmp(nw_X->gradient->buffer->data,
//                               X.grad().data_ptr(),
//                               X.grad().numel() * sizeof(float32_t)) == 0);

//         tensor_destroy(nw_X);
//         tensor_destroy(nw_Y);
//         tensor_destroy(nw_J);
//     }
// }
// END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Tensor Suite");
    tc_unary = tcase_create("Test Unary Tensor Case");
    tcase_add_checked_fixture(tc_unary, unary_setup, unary_teardown);
    tcase_add_test(tc_unary, test_exponential_forward);
    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_sample_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
