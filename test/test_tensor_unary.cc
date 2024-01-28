#include <iostream>
extern "C"
{
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <test_helper.h>
#include <random.h>
}
#include <test_helper_torch.h>

typedef enum tensor_unary_type_t
{
    TENSOR_EXPONENTIAL,
    TENSOR_LOGARITHM,
    TENSOR_SINE,
    TENSOR_COSINE,
    TENSOR_SQUARE_ROOT,
    TENSOR_RECIPROCAL,
    TENSOR_CONTIGUOUS,
    TENSOR_NEGATION,
    TENSOR_RECTIFIED_LINEAR,
    TENSOR_SIGMOID,
    TENSOR_LEAKY_RECTIFIED_LINEAR,
} tensor_unary_type_t;

#define CASES_0_0 7
#define CASES_1_0 10
#define CASES_2_0 16
#define CASES_3_0 24
#define CASES_4_0 32
#define CASES_5_0 30

#define CASES CASES_0_0 + CASES_1_0 + CASES_2_0 + CASES_3_0 + CASES_4_0 + CASES_5_0

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_gradients[RUNTIMES][DATATYPES][CASES];
torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    // Cases 0.0
    {},
    {1},
    {2},
    {3, 2},
    {2, 3, 4},
    {2, 3, 4, 5},
    {6, 5, 4, 3, 2},
    // Cases 1.0
    {},
    {1},
    {1},
    {1},
    {1},
    {1},
    {3},
    {4},
    {5},
    {6},
    // Cases 2.0
    {1, 1},
    {1, 3},
    {2, 1},
    {2, 3},
    {1, 1},
    {1, 3},
    {2, 1},
    {2, 3},
    {1, 1},
    {1, 3},
    {2, 1},
    {2, 3},
    {1, 1},
    {1, 3},
    {2, 1},
    {2, 3},
    // Cases 3.0
    {1, 1, 1},
    {4, 3, 2},
    {4, 3, 1},
    {1, 3, 2},
    {4, 1, 2},
    {1, 1, 2},
    {4, 1, 1},
    {1, 3, 1},
    {1, 1, 1},
    {4, 3, 2},
    {4, 3, 1},
    {1, 3, 2},
    {4, 1, 2},
    {1, 1, 2},
    {4, 1, 1},
    {1, 3, 1},
    {1, 1, 1},
    {4, 3, 2},
    {4, 3, 1},
    {1, 3, 2},
    {4, 1, 2},
    {1, 1, 2},
    {4, 1, 1},
    {1, 3, 1},
    // Cases 4.0
    {1, 1, 1, 1},
    {5, 4, 3, 2},
    {5, 4, 3, 1},
    {5, 4, 1, 2},
    {5, 1, 3, 2},
    {1, 4, 3, 2},
    {1, 1, 3, 2},
    {1, 4, 1, 2},
    {1, 4, 3, 1},
    {5, 1, 1, 2},
    {5, 1, 3, 1},
    {5, 4, 1, 1},
    {5, 1, 1, 1},
    {1, 1, 1, 2},
    {1, 4, 1, 1},
    {1, 1, 3, 1},
    {1, 1, 1, 1},
    {5, 4, 3, 2},
    {5, 4, 3, 1},
    {5, 4, 1, 2},
    {5, 1, 3, 2},
    {1, 4, 3, 2},
    {1, 1, 3, 2},
    {1, 4, 1, 2},
    {1, 4, 3, 1},
    {5, 1, 1, 2},
    {5, 1, 3, 1},
    {5, 4, 1, 1},
    {5, 1, 1, 1},
    {1, 1, 1, 2},
    {1, 4, 1, 1},
    {1, 1, 3, 1},
    // Cases 5.0
    {1, 1, 1, 1, 1},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 1},
    {6, 5, 4, 1, 2},
    {6, 5, 1, 3, 2},
    {6, 1, 4, 3, 2},
    {1, 5, 4, 3, 2},
    {1, 1, 4, 3, 2},
    {1, 5, 1, 3, 2},
    {1, 5, 4, 1, 2},
    {1, 5, 4, 3, 1},
    {6, 1, 1, 3, 2},
    {6, 1, 4, 1, 2},
    {6, 1, 4, 3, 1},
    {6, 5, 1, 1, 2},
    {6, 5, 1, 3, 1},
    {6, 5, 4, 1, 1},
    {6, 5, 1, 1, 1},
    {6, 1, 4, 1, 1},
    {1, 5, 4, 1, 1},
    {1, 5, 1, 3, 1},
    {1, 1, 4, 3, 1},
    {1, 1, 4, 1, 2},
    {1, 1, 1, 3, 2},
    {6, 1, 1, 1, 2},
    {1, 1, 1, 1, 2},
    {1, 1, 1, 3, 1},
    {1, 1, 4, 1, 1},
    {1, 5, 1, 1, 1},
    {6, 1, 1, 1, 1},
};

std::vector<int64_t> expanded_shapes[CASES] = {
    // Cases 0.0
    {},
    {1},
    {2},
    {3, 2},
    {2, 3, 4},
    {2, 3, 4, 5},
    {6, 5, 4, 3, 2},
    // Cases 1.0
    {1},
    {2},
    {2, 3},
    {2, 3, 4},
    {2, 3, 4, 5},
    {2, 3, 4, 5, 6},
    {2, 3},
    {2, 3, 4},
    {2, 3, 4, 5},
    {2, 3, 4, 5, 6},
    // Cases 2.0
    {2, 3},
    {2, 3},
    {2, 3},
    {2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {5, 4, 2, 3},
    {5, 4, 2, 3},
    {5, 4, 2, 3},
    {5, 4, 2, 3},
    {6, 5, 4, 2, 3},
    {6, 5, 4, 2, 3},
    {6, 5, 4, 2, 3},
    {6, 5, 4, 2, 3},
    // Cases 3.0
    {4, 3, 2},
    {4, 3, 2},
    {4, 3, 2},
    {4, 3, 2},
    {4, 3, 2},
    {4, 3, 2},
    {4, 3, 2},
    {4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    // Cases 4.0
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    // Cases 5.0
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
};

float32_t leaky_relu_arguments_f[CASES];
float64_t leaky_relu_arguments[CASES];

void setup(void)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; ++k)
            {
                tensors[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradients[i][j][k] = NULL;

                switch ((datatype_t) j)
                {
                case FLOAT32:
                    leaky_relu_arguments_f[k] = uniformf(0.0, 0.99);
                    torch_tensors[i][j][k] = torch::randn(shapes[k], 
                                                          torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
                                                          ).expand(expanded_shapes[k]);
                    break;
                case FLOAT64:
                    leaky_relu_arguments[k] = uniform(0.0, 0.99);
                    torch_tensors[i][j][k] = torch::randn(shapes[k],
                                                          torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)
                                                          ).expand(expanded_shapes[k]);
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors[i][j][k].retain_grad();

                tensors[i][j][k] = torch_to_tensor(torch_tensors[i][j][k], (runtime_t) i, (datatype_t) j);
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
                tensor_destroy(expected_gradients[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_unary(tensor_unary_type_t tensor_unary_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            datatype_t datatype = (datatype_t) j;
            for (int k = 0; k < CASES; k++)
            {
                torch::Tensor expected_tensor;

                switch (tensor_unary_type)
                {
                case TENSOR_EXPONENTIAL:
                    expected_tensor = torch::exp(torch_tensors[i][j][k]);
                    break;
                case TENSOR_LOGARITHM:
                    expected_tensor = torch::log(torch_tensors[i][j][k]);
                    break;
                case TENSOR_SINE:
                    expected_tensor = torch::sin(torch_tensors[i][j][k]);
                    break;
                case TENSOR_COSINE:
                    expected_tensor = torch::cos(torch_tensors[i][j][k]);
                    break;
                case TENSOR_SQUARE_ROOT:
                    expected_tensor = torch::sqrt(torch_tensors[i][j][k]);
                    break;
                case TENSOR_RECIPROCAL:
                    expected_tensor = torch::reciprocal(torch_tensors[i][j][k]);
                    break;
                case TENSOR_CONTIGUOUS:
                    expected_tensor = torch_tensors[i][j][k].contiguous();
                    break;
                case TENSOR_NEGATION:
                    expected_tensor = torch::neg(torch_tensors[i][j][k]);
                    break;
                case TENSOR_RECTIFIED_LINEAR:
                    expected_tensor = torch::relu(torch_tensors[i][j][k]);
                    break;
                case TENSOR_SIGMOID:
                    expected_tensor = torch::sigmoid(torch_tensors[i][j][k]);
                    break;
                case TENSOR_LEAKY_RECTIFIED_LINEAR:
                    switch (datatype)
                    {
                    case FLOAT32:
                        expected_tensor = torch::leaky_relu(torch_tensors[i][j][k], leaky_relu_arguments_f[k]);
                        break;
                    case FLOAT64:
                        expected_tensor = torch::leaky_relu(torch_tensors[i][j][k], leaky_relu_arguments[k]);
                        break;
                    default:
                        ck_abort_msg("unknown data type.");
                    }
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }
                expected_tensor.sum().backward();

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (tensor_unary_type)
                {
                case TENSOR_EXPONENTIAL:
                    error = tensor_exponential(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_LOGARITHM:
                    error = tensor_logarithm(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_SINE:
                    error = tensor_sine(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_COSINE:
                    error = tensor_cosine(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_SQUARE_ROOT:
                    error = tensor_square_root(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_RECIPROCAL:
                    error = tensor_reciprocal(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_CONTIGUOUS:
                    error = tensor_contiguous(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_NEGATION:
                    error = tensor_negation(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_RECTIFIED_LINEAR:
                    error = tensor_rectified_linear(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_SIGMOID:
                    error = tensor_sigmoid(tensors[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_LEAKY_RECTIFIED_LINEAR:
                    switch (datatype)
                    {
                    case FLOAT32:
                        error = tensor_leaky_rectified_linear(tensors[i][j][k], (void *) &leaky_relu_arguments_f[k], &returned_tensors[i][j][k]);
                        break;
                    case FLOAT64:
                        error = tensor_leaky_rectified_linear(tensors[i][j][k], (void *) &leaky_relu_arguments[k], &returned_tensors[i][j][k]);
                        break;
                    default:
                        ck_abort_msg("unknown data type.");
                    }
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }
                ck_assert_ptr_null(error);

                if (tensor_unary_type == TENSOR_CONTIGUOUS)
                {
                    ck_assert_tensor_eq(returned_tensors[i][j][k], expected_tensors[i][j][k]);
                }
                else
                {
                    ck_assert_tensor_equiv(returned_tensors[i][j][k], expected_tensors[i][j][k]);
                }

                expected_gradients[i][j][k] = torch_to_tensor(torch_tensors[i][j][k].grad(), (runtime_t) i, (datatype_t) j);

                // Back prop
                tensor_t *cost = NULL;
                error = tensor_summation(returned_tensors[i][j][k], &cost, NULL, 0, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors[i][j][k]->gradient, expected_gradients[i][j][k]);
            }
        }
    }
}

START_TEST(test_exponential)
{
    test_unary(TENSOR_EXPONENTIAL);
}
END_TEST

START_TEST(test_logarithm)
{
    test_unary(TENSOR_LOGARITHM);
}
END_TEST

START_TEST(test_sine)
{
    test_unary(TENSOR_SINE);
}
END_TEST

START_TEST(test_cosine)
{
    test_unary(TENSOR_COSINE);
}
END_TEST

START_TEST(test_square_root)
{
    test_unary(TENSOR_SQUARE_ROOT);
}
END_TEST

START_TEST(test_reciprocal)
{
    test_unary(TENSOR_RECIPROCAL);
}
END_TEST

START_TEST(test_contiguous)
{
    test_unary(TENSOR_CONTIGUOUS);
}
END_TEST

START_TEST(test_negation)
{
    test_unary(TENSOR_NEGATION);
}
END_TEST

START_TEST(test_rectified_linear)
{
    test_unary(TENSOR_RECTIFIED_LINEAR);
}
END_TEST

START_TEST(test_sigmoid)
{
    test_unary(TENSOR_SIGMOID);
}
END_TEST

START_TEST(test_leaky_rectified_linear)
{
    test_unary(TENSOR_LEAKY_RECTIFIED_LINEAR);
}
END_TEST

Suite *make_unary_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Unary Tensor Suite");

    tc_unary = tcase_create("Test Unary Tensor Case");
    tcase_add_checked_fixture(tc_unary, setup, teardown);
    tcase_add_test(tc_unary, test_exponential);
    tcase_add_test(tc_unary, test_logarithm);
    tcase_add_test(tc_unary, test_sine);
    tcase_add_test(tc_unary, test_cosine);
    tcase_add_test(tc_unary, test_square_root);
    tcase_add_test(tc_unary, test_reciprocal);
    tcase_add_test(tc_unary, test_contiguous);
    tcase_add_test(tc_unary, test_negation);
    tcase_add_test(tc_unary, test_rectified_linear);
    tcase_add_test(tc_unary, test_sigmoid);
    tcase_add_test(tc_unary, test_leaky_rectified_linear);

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);
    set_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_unary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
