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
}
#include <test_helper_torch.h>

#define CASES 3

std::vector<int64_t> shapes_w[CASES] = {
    {5, 3, 6, 7},
    {1, 1, 9, 7},
    {10, 9, 9, 11},
};

std::vector<int64_t> shapes_x[CASES] = {
    {5, 3, 3, 3},
    {10, 1, 4, 4},
    {11, 9, 1, 1},
};

std::vector<int64_t> shapes_y[CASES] = {
    {5},
    {10},
    {11},
};

int64_t stride[CASES] = {
    2,
    1,
    3,
};

int64_t padding[CASES] = {
    1,
    3,
    0,
};

nw_error_t *error = NULL;

std::vector<tensor_t *> tensors_w[RUNTIMES][DATATYPES];
std::vector<tensor_t *> tensors_x[RUNTIMES][DATATYPES];
std::vector<tensor_t *> tensors_y[RUNTIMES][DATATYPES];
std::vector<tensor_t *> returned_tensors[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_tensors[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_w[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_x[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_y[RUNTIMES][DATATYPES];

std::vector<torch::Tensor> torch_tensors_w[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_tensors_x[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_tensors_y[RUNTIMES][DATATYPES];

typedef enum tensor_ternary_operation_type_t
{
    TENSOR_CONVOLUTION,
} tensor_reduction_operation_type_t;

void setup(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            tensors_w[i][j] = std::vector<tensor_t *>(CASES);
            tensors_x[i][j] = std::vector<tensor_t *>(CASES);
            tensors_y[i][j] = std::vector<tensor_t *>(CASES);
            returned_tensors[i][j] = std::vector<tensor_t *>(CASES);
            expected_tensors[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_w[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_x[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_y[i][j] = std::vector<tensor_t *>(CASES);
            torch_tensors_w[i][j] = std::vector<torch::Tensor>(CASES);
            torch_tensors_x[i][j] = std::vector<torch::Tensor>(CASES);
            torch_tensors_y[i][j] = std::vector<torch::Tensor>(CASES);

            for (int k = 0; k < CASES; ++k)
            {
                tensors_w[i][j][k] = NULL;
                tensors_x[i][j][k] = NULL;
                tensors_y[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradients_w[i][j][k] = NULL;
                expected_gradients_x[i][j][k] = NULL;
                expected_gradients_y[i][j][k] = NULL;

                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors_w[i][j][k] = torch::randn(shapes_w[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    break;
                case FLOAT64:
                    torch_tensors_w[i][j][k] = torch::randn(shapes_w[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors_w[i][j][k].retain_grad();
                torch_tensors_x[i][j][k].retain_grad();
                torch_tensors_y[i][j][k].retain_grad();

                tensors_w[i][j][k] = torch_to_tensor(torch_tensors_w[i][j][k], (runtime_t) i, (datatype_t) j);
                tensors_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k], (runtime_t) i, (datatype_t) j);
                tensors_y[i][j][k] = torch_to_tensor(torch_tensors_y[i][j][k], (runtime_t) i, (datatype_t) j);
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
                tensor_destroy(tensors_w[i][j][k]);
                tensor_destroy(tensors_x[i][j][k]);
                tensor_destroy(tensors_y[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
                tensor_destroy(expected_gradients_w[i][j][k]);
                tensor_destroy(expected_gradients_x[i][j][k]);
                tensor_destroy(expected_gradients_y[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_ternary(tensor_ternary_operation_type_t tensor_ternary_operation_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;

                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION:
                    expected_tensor = torch::nn::functional::conv2d(torch_tensors_w[i][j][k], 
                                                                    torch_tensors_x[i][j][k], 
                                                                    torch::nn::functional::Conv2dFuncOptions().padding(padding[k])
                                                                                                              .stride(stride[k])
                                                                                                              .bias(torch_tensors_y[i][j][k]));
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION:
                    error = tensor_convolution(tensors_w[i][j][k], tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k], stride[k], padding[k]);
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(returned_tensors[i][j][k], expected_tensors[i][j][k]);

                expected_tensor.sum().backward();
                expected_gradients_w[i][j][k] = torch_to_tensor(torch_tensors_w[i][j][k].grad(), (runtime_t) i, (datatype_t) j);
                expected_gradients_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k].grad(), (runtime_t) i, (datatype_t) j);
                expected_gradients_y[i][j][k] = torch_to_tensor(torch_tensors_y[i][j][k].grad(), (runtime_t) i, (datatype_t) j);

                tensor_t *cost = NULL;
                error = tensor_summation(returned_tensors[i][j][k], &cost, NULL, 0, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors_w[i][j][k]->gradient, expected_gradients_w[i][j][k]);
                ck_assert_tensor_equiv(tensors_x[i][j][k]->gradient, expected_gradients_x[i][j][k]);
                ck_assert_tensor_equiv(tensors_y[i][j][k]->gradient, expected_gradients_y[i][j][k]);
            }
        }
    }
}

START_TEST(test_tensor_convolution)
{
    test_ternary(TENSOR_CONVOLUTION);
}
END_TEST

Suite *make_ternary_suite(void)
{
    Suite *s;
    TCase *tc;

    s = suite_create("Test Ternary Tensor Suite");

    tc = tcase_create("Test Ternary Case");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, test_tensor_convolution);
    suite_add_tcase(s, tc);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_ternary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}