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

#define CASES 9

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

view_t *views[RUNTIMES][DATATYPES][CASES];
view_t *expected_views[RUNTIMES][DATATYPES][CASES];

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> axis[CASES] = {
    {0},
    {0},
    {1},
    {1},
    {0, 2},
    {1},
    {0, 1, 2, 3},
    {0, 1, 2, 3},
    {0, 2, 3},
};

uint64_t length[CASES] = {
    1,
    1,
    1,
    1,
    2,
    1,
    4,
    4,
    3,
};

bool_t keep_dimension[CASES] = {
    false,
    true,
    false,
    true,
    true,
    false,
    false,
    false,
    true,
};

std::vector<int64_t> torch_shapes[CASES] = {
    {1},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
    {5},
    {3, 1, 1},
};

std::vector<int64_t> shapes[CASES] = {
    {2},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
    {2, 3, 4, 5},
    {2, 3, 4, 5},
};

uint64_t expected_shapes[][CASES] = {
    {},
    {1},
    {10},
    {10, 1},
    {1, 4, 1},
    {3, 5},
    {},
    {},
    {1, 3, 1, 1},
};

uint64_t ranks[CASES] = {
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    4,
};

uint64_t expected_ranks[CASES] = {
    0,
    1,
    1,
    2,
    3,
    2,
    0,
    0,
    4,
};

uint64_t strides[][CASES] = {
    {0},
    {1},
    {1, 0},
    {10, 1},
    {20, 5, 1},
    {20, 5, 1},
    {60, 20, 5, 1},
    {0, 0, 0, 1},
    {0, 1, 0, 0},
};

uint64_t expected_strides[][CASES] = {
    {},
    {0},
    {1},
    {1, 0},
    {0, 1, 0},
    {5, 1},
    {},
    {},
    {0, 1, 0, 0},
};

uint64_t n[CASES] = {
    1,
    10,
    10,
    100,
    60,
    60,
    120,
    5,
    3,
};

uint64_t expected_n[CASES] = {
    1,
    1,
    10,
    10,
    4,
    15,
    1,
    1,
    3,
};

void setup(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
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
                                    ranks[k],
                                    (uint64_t *) shapes[k].data(),
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

                error = tensor_create(&tensors[i][j][k], buffers[i][j][k], NULL, NULL, false, false);
                ck_assert_ptr_null(error);

                error = tensor_create_empty(&expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) torch_tensors[i][j][k].storage_offset(),
                                    expected_ranks[k],
                                    expected_shapes[k],
                                    expected_strides[k]);
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

START_TEST(test_summation_forward)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                printf("test_summation_forward:runtime %s:datatype %d:case %d\n", 
                       runtime_string((runtime_t) i), j, k);

                function_t *function = NULL; 
                reduction_operation_t *reduction_operation = NULL;
                operation_t *operation = NULL;

                torch::Tensor expected_tensor = torch::sum(torch_tensors[i][j][k], 
                                                           axis[k],
                                                           keep_dimension[k]);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      expected_n[k],
                                      true);
                ck_assert_ptr_null(error);


                error = reduction_operation_create(&reduction_operation,
                                                   SUMMATION_OPERATION,
                                                   tensors[i][j][k],
                                                   (uint64_t *) axis[k].data(),
                                                   length[k], 
                                                   keep_dimension[k],
                                                   expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);
                error = operation_create(&operation, REDUCTION_OPERATION, reduction_operation);
                ck_assert_ptr_null(error);
                error = function_create(&function, operation, REDUCTION_OPERATION);
                ck_assert_ptr_null(error);

                expected_tensors[i][j][k]->buffer = expected_buffers[i][j][k];
                expected_tensors[i][j][k]->context = function;
                expected_tensors[i][j][k]->requires_gradient = tensors[i][j][k]->requires_gradient;
                expected_tensors[i][j][k]->lock = tensors[i][j][k]->lock;

                error = tensor_summation(tensors[i][j][k], 
                                         returned_tensors[i][j][k],
                                         (uint64_t *) axis[k].data(),
                                         length[k],
                                         keep_dimension[k]);
                ck_assert_ptr_null(error);

                ck_assert_tensor_eq(returned_tensors[i][j][k], expected_tensors[i][j][k]);
            }
        }
    }
}
END_TEST

Suite *make_reduction_suite(void)
{
    Suite *s;
    TCase *tc_reduction;

    s = suite_create("Test Reduction Tensor Suite");

    tc_reduction = tcase_create("Test Reduction Tensor Case");
    tcase_add_checked_fixture(tc_reduction, setup, teardown);
    tcase_add_test(tc_reduction, test_summation_forward);

    suite_add_tcase(s, tc_reduction);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_reduction_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}