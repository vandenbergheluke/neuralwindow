#include <iostream>
extern "C"
{
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <graph.h>
#include <cost.h>
#include <test_helper.h>
}
#include <test_helper_torch.h>

#define CASES 1
#define LAYERS 5

nw_error_t *error;

tensor_t *input[RUNTIMES][DATATYPES][CASES];
tensor_t *output[RUNTIMES][DATATYPES][CASES];
tensor_t *cost[RUNTIMES][DATATYPES][CASES];
tensor_t *weights[RUNTIMES][DATATYPES][CASES][LAYERS];
tensor_t *bias[RUNTIMES][DATATYPES][CASES][LAYERS];

tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES][LAYERS];
tensor_t *returned_tensors_i[RUNTIMES][DATATYPES][CASES][LAYERS];
tensor_t *returned_tensors_j[RUNTIMES][DATATYPES][CASES][LAYERS];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES][LAYERS];
tensor_t *expected_gradients_weights[RUNTIMES][DATATYPES][CASES][LAYERS];
tensor_t *expected_gradients_bias[RUNTIMES][DATATYPES][CASES][LAYERS];

torch::Tensor torch_input[RUNTIMES][DATATYPES][CASES];
torch::Tensor torch_output[RUNTIMES][DATATYPES][CASES];
torch::Tensor torch_weights[RUNTIMES][DATATYPES][CASES][LAYERS];
torch::Tensor torch_bias[RUNTIMES][DATATYPES][CASES][LAYERS];

std::vector<int64_t> input_shape[CASES] = {
    {5, 10},
};

std::vector<int64_t> output_shape[CASES] = {
    {5, 1},
};

std::vector<int64_t> weight_shapes[CASES][LAYERS] = {
    {
        {10, 11},
        {11, 12},
        {12, 11},
        {11, 10},
        {10, 10},
    },
};

std::vector<int64_t> bias_shapes[CASES][LAYERS] = {
    {
        {11},
        {12},
        {11},
        {10},
        {10},
    },
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
                input[i][j][k] = NULL;
                output[i][j][k] = NULL;
                cost[i][j][k] = NULL;

                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_input[i][j][k] = torch::randn(input_shape[k],
                                                    torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .requires_grad(false));
                    torch_output[i][j][k] = torch::randint(0, 10, output_shape[k],
                                                           torch::TensorOptions()
                                                          .dtype(torch::kFloat32)
                                                         .requires_grad(false));
                    break;
                case FLOAT64:
                    torch_input[i][j][k] = torch::randn(input_shape[k],
                                                    torch::TensorOptions()
                                                    .dtype(torch::kFloat64)
                                                    .requires_grad(false));
                    torch_output[i][j][k] = torch::randint(0, 10, output_shape[k],
                                                           torch::TensorOptions()
                                                          .dtype(torch::kFloat64)
                                                         .requires_grad(false));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                input[i][j][k] = torch_to_tensor(torch_input[i][j][k], (runtime_t) i, (datatype_t) j);
                output[i][j][k] = torch_to_tensor(torch_output[i][j][k], (runtime_t) i, (datatype_t) j);

                ck_assert_ptr_null(error);

                for (int l = 0; l < LAYERS; ++l)
                {
                    returned_tensors[i][j][k][l] = NULL;
                    returned_tensors_i[i][j][k][l] = NULL;
                    returned_tensors_j[i][j][k][l] = NULL;
                    expected_tensors[i][j][k][l] = NULL;
                    weights[i][j][k][l] = NULL;
                    bias[i][j][k][l] = NULL;
                    expected_gradients_weights[i][j][k][l] = NULL;
                    expected_gradients_bias[i][j][k][l] = NULL;

                    switch ((datatype_t) j)
                    {
                    case FLOAT32:
                        torch_weights[i][j][k][l] = torch::randn(weight_shapes[k][l],
                                                                 torch::TensorOptions()
                                                                 .dtype(torch::kFloat32)
                                                                 .requires_grad(true));
                        torch_bias[i][j][k][l] = torch::randn(bias_shapes[k][l],
                                                              torch::TensorOptions().
                                                              dtype(torch::kFloat32).
                                                              requires_grad(true));
                        break;
                    case FLOAT64:
                        torch_weights[i][j][k][l] = torch::randn(weight_shapes[k][l],
                                                                 torch::TensorOptions()
                                                                 .dtype(torch::kFloat64)
                                                                 .requires_grad(true));
                        torch_bias[i][j][k][l] = torch::randn(bias_shapes[k][l],
                                                              torch::TensorOptions().
                                                              dtype(torch::kFloat64).
                                                              requires_grad(true));
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }
                    torch_weights[i][j][k][l].retain_grad();
                    torch_bias[i][j][k][l].retain_grad();

                    weights[i][j][k][l] = torch_to_tensor(torch_weights[i][j][k][l], (runtime_t) i, (datatype_t) j);
                    bias[i][j][k][l] = torch_to_tensor(torch_bias[i][j][k][l], (runtime_t) i, (datatype_t) j);
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
                tensor_destroy(input[i][j][k]);
                tensor_destroy(output[i][j][k]);
                for (int l = 0; l < LAYERS; l++)
                {
                    tensor_destroy(expected_tensors[i][j][k][l]);
                    tensor_destroy(weights[i][j][k][l]);
                    tensor_destroy(bias[i][j][k][l]);
                    tensor_destroy(expected_gradients_weights[i][j][k][l]);
                    tensor_destroy(expected_gradients_bias[i][j][k][l]);

                }
            }
        }
    }
    error_print(error);
    error_destroy(error);
}

START_TEST(test_feed_forward_neural_network)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                torch::Tensor expected_tensor = torch_input[i][j][k];
                start_graph();
                for (int l = 0; l < LAYERS; l++)
                {
                    expected_tensor = torch::matmul(expected_tensor, torch_weights[i][j][k][l]);
                    expected_tensor = torch::add(expected_tensor, torch_bias[i][j][k][l]);
                    if (l != LAYERS - 1)
                    {
                        expected_tensor = torch::relu(expected_tensor);
                    }
                    if (!l)
                    {
                        error = tensor_matrix_multiplication(input[i][j][k], weights[i][j][k][l], &returned_tensors_i[i][j][k][l]);
                    }
                    else
                    {
                        error = tensor_matrix_multiplication(returned_tensors[i][j][k][l - 1], weights[i][j][k][l], &returned_tensors_i[i][j][k][l]);
                    }
                    ck_assert_ptr_null(error);

                    error = tensor_addition(returned_tensors_i[i][j][k][l], bias[i][j][k][l], &returned_tensors_j[i][j][k][l]);
                    ck_assert_ptr_null(error);

                    if (l == LAYERS - 1)
                    {
                        returned_tensors[i][j][k][l] = returned_tensors_j[i][j][k][l];
                    }
                    else
                    {
                        error = tensor_rectified_linear(returned_tensors_j[i][j][k][l], &returned_tensors[i][j][k][l]);
                    }
                    ck_assert_ptr_null(error);
                    
                    expected_tensors[i][j][k][l] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                    ck_assert_tensor_equiv(returned_tensors[i][j][k][l], expected_tensors[i][j][k][l]);
                }
                
                // Backward Propogation
                torch::cross_entropy_loss(expected_tensor, torch_output[i][j][k].to(torch::kLong).view(-1)).backward();
                error = categorical_cross_entropy(output[i][j][k], returned_tensors[i][j][k][LAYERS - 1], &cost[i][j][k]);
                ck_assert_ptr_null(error);
                end_graph();
                error = tensor_backward(cost[i][j][k], NULL);
                ck_assert_ptr_null(error);

                for (int l = 0; l < LAYERS; l++)
                {
                    expected_gradients_weights[i][j][k][l] = torch_to_tensor(torch_weights[i][j][k][l].grad(), (runtime_t) i, (datatype_t) j);
                    expected_gradients_bias[i][j][k][l] = torch_to_tensor(torch_bias[i][j][k][l].grad(), (runtime_t) i, (datatype_t) j);

                    ck_assert_tensor_equiv(weights[i][j][k][l]->gradient, expected_gradients_weights[i][j][k][l]);
                    ck_assert_tensor_equiv(bias[i][j][k][l]->gradient, expected_gradients_bias[i][j][k][l]);
                }
            }
        }
    }
}
END_TEST

Suite *make_binary_suite(void)
{
    Suite *s;
    TCase *tc_autograd;

    s = suite_create("Test Autograd Suite");

    tc_autograd= tcase_create("Test Autograd Case");
    tcase_add_checked_fixture(tc_autograd, setup, teardown);
    tcase_add_test(tc_autograd, test_feed_forward_neural_network);

    suite_add_tcase(s, tc_autograd);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    const char *env_variable = "GRAPH=1";
    if (putenv((char *)env_variable) != 0) {
        return EXIT_FAILURE;
    }

    sr = srunner_create(make_binary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}