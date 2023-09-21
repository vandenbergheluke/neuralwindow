#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <datatype.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
}
#include <test_helper.h>

typedef enum tensor_reduction_type_t
{
    TENSOR_SUMMATION,
    TENSOR_MAXIMUM,
    TENSOR_MEAN,
    TENSOR_SOFTMAX,
    TENSOR_ARGUMENT_MAXIMUM,
} tensor_reduction_type_t;

#define CASES 9
#define KEEP_DIMENSIONS 2

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES][KEEP_DIMENSIONS];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES][KEEP_DIMENSIONS];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES][KEEP_DIMENSIONS];
tensor_t *expected_gradient[RUNTIMES][DATATYPES][CASES][KEEP_DIMENSIONS];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES][KEEP_DIMENSIONS];

std::vector<int64_t> axis[CASES] = {
    {},
    {},
    {},
    {0},
    {1},
    {},
    {0},
    {0, 1},
    {1, 0},
};

std::vector<int64_t> shapes[CASES] = {
    {},
    {1},
    {2},
    {2},
    {1, 2},
    {1, 2},
    {1, 2},
    {1, 2},
    {1, 2},
};

std::vector<int64_t> expanded_shapes[CASES] = {
    {},
    {1},
    {2},
    {2},
    {1, 2},
    {1, 2},
    {1, 2},
    {1, 2},
    {1, 2},
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
                for (int l = 0; l < KEEP_DIMENSIONS; ++l)
                {
                    tensors[i][j][k][l] = NULL;
                    returned_tensors[i][j][k][l] = NULL;
                    expected_tensors[i][j][k][l] = NULL;
                    expected_gradient[i][j][k][l] = NULL;

                    switch ((datatype_t) j)
                    {
                    case FLOAT32:
                        torch_tensors[i][j][k][l] = torch::randn(shapes[k], 
                                                                 torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
                                                                 ).expand(expanded_shapes[k]);
                        break;
                    case FLOAT64:
                        torch_tensors[i][j][k][l] = torch::randn(shapes[k],
                                                                 torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)
                                                                 ).expand(expanded_shapes[k]);
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }
                    torch_tensors[i][j][k][l].retain_grad();

                    tensors[i][j][k][l] = torch_to_tensor(torch_tensors[i][j][k][l], (runtime_t) i, (datatype_t) j);
                }
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
                for (int l = 0; l < KEEP_DIMENSIONS; l++)
                {
                    tensor_destroy(tensors[i][j][k][l]);
                    tensor_destroy(expected_tensors[i][j][k][l]);
                    tensor_destroy(expected_gradient[i][j][k][l]);
                }
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_reduction(tensor_reduction_type_t tensor_reduction_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                for (int l = 0; l < KEEP_DIMENSIONS; l++)
                {
                    torch::Tensor expected_tensor;

                    switch (tensor_reduction_type)
                    {
                    case TENSOR_SUMMATION:
                        expected_tensor = torch::sum(torch_tensors[i][j][k][l], axis[k], (bool_t) l);
                        break;
                    case TENSOR_MAXIMUM:
                        expected_tensor = torch::amax(torch_tensors[i][j][k][l], axis[k], (bool_t) l);
                        break;
                    case TENSOR_MEAN:
                        expected_tensor = torch::mean(torch_tensors[i][j][k][l], axis[k],  (bool_t) l);
                        break;
                    case TENSOR_SOFTMAX:
                        expected_tensor = torch::softmax(torch_tensors[i][j][k][l], (axis[k].size()) ? axis[k][0] : 0);
                        break;
                    case TENSOR_ARGUMENT_MAXIMUM:
                        expected_tensor = torch::argmax(torch_tensors[i][j][k][l], (axis[k].size()) ? axis[k][0] : 0, (bool_t) l);
                        break;
                    default:
                        ck_abort_msg("unknown reduction type.");
                    }

                    expected_tensors[i][j][k][l] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                    switch (tensor_reduction_type)
                    {
                    case TENSOR_SUMMATION:
                        error = tensor_summation(tensors[i][j][k][l], 
                                                 &returned_tensors[i][j][k][l],
                                                 (uint64_t *) axis[k].data(),
                                                 (uint64_t) axis[k].size(),
                                                 (bool_t) l);
                        break;
                    case TENSOR_MAXIMUM:
                        error = tensor_maximum(tensors[i][j][k][l], 
                                               &returned_tensors[i][j][k][l],
                                               (uint64_t *) axis[k].data(),
                                               (uint64_t) axis[k].size(),
                                               (bool_t) l);
                        break;
                    case TENSOR_MEAN:
                        error = tensor_mean(tensors[i][j][k][l], 
                                            &returned_tensors[i][j][k][l],
                                            (uint64_t *) axis[k].data(),
                                            (uint64_t) axis[k].size(),
                                            (bool_t) l);
                        break;
                    case TENSOR_SOFTMAX:
                        error = tensor_softmax(tensors[i][j][k][l], 
                                               &returned_tensors[i][j][k][l],
                                               (axis[k].size()) ? *(uint64_t *) axis[k].data() : (uint64_t) 0);
                        break;
                    case TENSOR_ARGUMENT_MAXIMUM:
                        error = tensor_argument_maximum(tensors[i][j][k][l], 
                                                        &returned_tensors[i][j][k][l],
                                                        (axis[k].size()) ? *(uint64_t *) axis[k].data() : (uint64_t) 0,
                                                        (bool_t) l);
                        break;
                    default:
                        ck_abort_msg("unknown reduction type.");
                    }
                    ck_assert_ptr_null(error);

                    ck_assert_tensor_equiv(returned_tensors[i][j][k][l],
                                           expected_tensors[i][j][k][l]);

                    if (tensor_reduction_type == TENSOR_ARGUMENT_MAXIMUM)
                    {
                        tensor_destroy(returned_tensors[i][j][k][l]);
                        continue;
                    }

                    // Back prop
                    expected_tensor.sum().backward();
                    expected_gradient[i][j][k][l] = torch_to_tensor(torch_tensors[i][j][k][l].grad(), (runtime_t) i, (datatype_t) j);
                    tensor_t *cost;
                    error = tensor_summation(returned_tensors[i][j][k][l], &cost, NULL, 0, false);
                    ck_assert_ptr_null(error);
                    error = tensor_backward(cost, NULL);
                    ck_assert_ptr_null(error);

                    ck_assert_tensor_equiv(tensors[i][j][k][l]->gradient, expected_gradient[i][j][k][l]);
                }
            }
        }
    }
}

START_TEST(test_summation)
{
    test_reduction(TENSOR_SUMMATION);
}
END_TEST

START_TEST(test_maximum)
{
    test_reduction(TENSOR_MAXIMUM);
}
END_TEST

START_TEST(test_mean)
{
    test_reduction(TENSOR_MEAN);
}
END_TEST

START_TEST(test_softmax)
{
    test_reduction(TENSOR_SOFTMAX);
}
END_TEST

START_TEST(test_argument_maximum)
{
    test_reduction(TENSOR_ARGUMENT_MAXIMUM);
}
END_TEST

Suite *make_reduction_suite(void)
{
    Suite *s;
    TCase *tc_reduction;

    s = suite_create("Test Reduction Tensor Suite");

    tc_reduction = tcase_create("Test Reduction Tensor Case");
    tcase_add_checked_fixture(tc_reduction, setup, teardown);
    tcase_add_test(tc_reduction, test_summation);
    tcase_add_test(tc_reduction, test_maximum);
    tcase_add_test(tc_reduction, test_mean);
    tcase_add_test(tc_reduction, test_softmax);
    tcase_add_test(tc_reduction, test_argument_maximum);

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