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
#include <test_helper.h>
}
#include <torch/torch.h>
#include <cstring>

#define CASES 7

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

view_t *views[RUNTIMES][DATATYPES][CASES];
view_t *expected_views[RUNTIMES][DATATYPES][CASES];

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> torch_shapes[CASES] = {
    {1},
    {1},
    {1},
    {1},
    {3, 4, 5},
    {3, 4, 5},
    {1, 2, 3, 4, 5},
};

std::vector<int64_t> shapes[CASES] = {
    {1},
    {1},
    {1},
    {1},
    {3, 4, 5},
    {3, 4, 5},
    {1, 2, 3, 4, 5},
};

uint64_t strides[][CASES] = {
    {0},
    {0},
    {0},
    {0},
    {20, 5, 1},
    {0, 0, 0},
    {0, 0, 4, 1, 0},
};

uint64_t n[CASES] = {
    1,
    1,
    1,
    1,
    60,
    1,
    12,
};

void setup(void)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; ++k)
            {
                buffers[i][j][k] = NULL;
                expected_buffers[i][j][k] = NULL;

                views[i][j][k] = NULL;
                expected_views[i][j][k] = NULL;

                tensors[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
            }

            
            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors[i][j][k] = torch::randn(torch_shapes[k], 
                                                          torch::TensorOptions().dtype(torch::kFloat32)
                                                          ).expand(shapes[k]);
                    break;
                case FLOAT64:
                    torch_tensors[i][j][k] = torch::randn(torch_shapes[k],
                                                          torch::TensorOptions().dtype(torch::kFloat64)
                                                          ).expand(shapes[k]);
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                error = view_create(&views[i][j][k], 
                                    (uint64_t) torch_tensors[i][j][k].storage_offset(),
                                    (uint64_t) torch_tensors[i][j][k].ndimension(),
                                    (uint64_t *) torch_tensors[i][j][k].sizes().data(),
                                    strides[k]);
                ck_assert_ptr_null(error);

                error = buffer_create(&buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      views[i][j][k],
                                      (void *) torch_tensors[i][j][k].data_ptr(),
                                      n[k],
                                      true);
                ck_assert_ptr_null(error);

                error = tensor_create(&tensors[i][j][k],
                                      buffers[i][j][k],
                                      NULL,
                                      NULL,
                                      false,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create_empty(&expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);

                error = tensor_create_empty(&returned_tensors[i][j][k]);
                ck_assert_ptr_null(error);
            }
        }
    }
}

void teardown(void)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                tensor_destroy(tensors[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
                tensor_destroy(returned_tensors[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

START_TEST(test_exponential_forward)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                function_t *function = NULL; 
                unary_operation_t *unary_operation = NULL;
                operation_t *operation = NULL;

                torch::Tensor expected_tensor = torch::exp(torch_tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    0,
                                    tensors[i][j][k]->buffer->view->rank,
                                    tensors[i][j][k]->buffer->view->shape,
                                    tensors[i][j][k]->buffer->view->strides);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      buffers[i][j][k]->n,
                                      true);
                ck_assert_ptr_null(error);

                error = unary_operation_create(&unary_operation,
                                               EXPONENTIAL_OPERATION,
                                               tensors[i][j][k],
                                               expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);
                error = operation_create(&operation, UNARY_OPERATION, unary_operation);
                ck_assert_ptr_null(error);
                error = function_create(&function, operation, UNARY_OPERATION);
                ck_assert_ptr_null(error);

                expected_tensors[i][j][k]->buffer = expected_buffers[i][j][k];
                expected_tensors[i][j][k]->context = function;
                expected_tensors[i][j][k]->requires_gradient = tensors[i][j][k]->requires_gradient;
                expected_tensors[i][j][k]->lock = tensors[i][j][k]->lock;

                error = tensor_exponential(tensors[i][j][k], returned_tensors[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_tensor_eq(returned_tensors[i][j][k], expected_tensors[i][j][k]);
            }
        }
    }
}
END_TEST

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
//                               runtimes[k],
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

Suite *make_unary_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Unary Tensor Suite");

    tc_unary = tcase_create("Test Unary Tensor Case");
    tcase_add_checked_fixture(tc_unary, setup, teardown);
    tcase_add_test(tc_unary, test_exponential_forward);

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_unary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
