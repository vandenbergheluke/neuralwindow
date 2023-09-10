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
#include <torch/torch.h>

#define CASES 1
#define LAYERS 5

nw_error_t *error;

tensor_t *input[RUNTIMES][DATATYPES][CASES];
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
torch::Tensor torch_weights[RUNTIMES][DATATYPES][CASES][LAYERS];
torch::Tensor torch_bias[RUNTIMES][DATATYPES][CASES][LAYERS];

std::vector<int64_t> input_shape[CASES] = {
    {5, 10},
};

std::vector<int64_t> weight_shapes[CASES][LAYERS] = {
    {
        {10, 11},
        {11, 12},
        {12, 11},
        {11, 10},
        {10, 1},
    },
};

std::vector<int64_t> bias_shapes[CASES][LAYERS] = {
    {
        {11},
        {12},
        {11},
        {10},
        {1},
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
                cost[i][j][k] = NULL;
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
                }
            }

            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_input[i][j][k] = torch::randn(input_shape[k],
                                                    torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .requires_grad(false));
                    break;
                case FLOAT64:
                    torch_input[i][j][k] = torch::randn(input_shape[k],
                                                    torch::TensorOptions()
                                                    .dtype(torch::kFloat64)
                                                    .requires_grad(false));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                view_t *view;
                storage_t *storage;
                buffer_t *buffer;
                
                error = view_create(&view, 
                                    (uint64_t) torch_input[i][j][k].storage_offset(),
                                    (uint64_t) torch_input[i][j][k].ndimension(),
                                    (uint64_t *) torch_input[i][j][k].sizes().data(),
                                    (uint64_t *) torch_input[i][j][k].strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_input[i][j][k].storage().nbytes() /
                                       (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_input[i][j][k].data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&input[i][j][k], buffer, NULL, NULL, false, true);
                ck_assert_ptr_null(error);

                error = tensor_create_default(&cost[i][j][k]);
                ck_assert_ptr_null(error);

                for (int l = 0; l < LAYERS; ++l)
                {
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

                    // Weights
                    error = view_create(&view, 
                                        (uint64_t) torch_weights[i][j][k][l].storage_offset(),
                                        (uint64_t) torch_weights[i][j][k][l].ndimension(),
                                        (uint64_t *) torch_weights[i][j][k][l].sizes().data(),
                                        (uint64_t *) torch_weights[i][j][k][l].strides().data());
                    ck_assert_ptr_null(error);

                    error = storage_create(&storage,
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) torch_weights[i][j][k][l].storage().nbytes() /
                                           (uint64_t) datatype_size((datatype_t) j),
                                           (void *) torch_weights[i][j][k][l].data_ptr());
                    ck_assert_ptr_null(error);

                    error = buffer_create(&buffer,
                                          view,
                                          storage,
                                          false);
                    ck_assert_ptr_null(error);

                    error = tensor_create(&weights[i][j][k][l], buffer, NULL, NULL, true, true);
                    ck_assert_ptr_null(error);

                    // Bias
                    error = view_create(&view, 
                                        (uint64_t) torch_bias[i][j][k][l].storage_offset(),
                                        (uint64_t) torch_bias[i][j][k][l].ndimension(),
                                        (uint64_t *) torch_bias[i][j][k][l].sizes().data(),
                                        (uint64_t *) torch_bias[i][j][k][l].strides().data());
                    ck_assert_ptr_null(error);

                    error = storage_create(&storage,
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) torch_bias[i][j][k][l].storage().nbytes() /
                                           (uint64_t) datatype_size((datatype_t) j),
                                           (void *) torch_bias[i][j][k][l].data_ptr());
                    ck_assert_ptr_null(error);

                    error = buffer_create(&buffer,
                                        view,
                                        storage,
                                        false);
                    ck_assert_ptr_null(error);

                    error = tensor_create(&bias[i][j][k][l], buffer, NULL, NULL, true, true);
                    ck_assert_ptr_null(error);

                    error = tensor_create_default(&returned_tensors[i][j][k][l]);
                    ck_assert_ptr_null(error);
                    returned_tensors[i][j][k][l]->lock = true;

                    error = tensor_create_default(&returned_tensors_i[i][j][k][l]);
                    ck_assert_ptr_null(error);
                    returned_tensors_i[i][j][k][l]->lock = true;
                    
                    error = tensor_create_default(&returned_tensors_j[i][j][k][l]);
                    ck_assert_ptr_null(error);
                    returned_tensors_j[i][j][k][l]->lock = true;
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
                for (int l = 0; l < LAYERS; l++)
                {
                    tensor_destroy(returned_tensors[i][j][k][l]);
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
                view_t *view;
                storage_t *storage;
                buffer_t *buffer;
                torch::Tensor expected_tensor = torch_input[i][j][k];
                for (int l = 0; l < LAYERS; l++)
                {
                    expected_tensor = torch::matmul(expected_tensor, 
                                                    torch_weights[i][j][k][l]);
                    expected_tensor = torch::add(expected_tensor,
                                                 torch_bias[i][j][k][l]);
                    if (l == LAYERS - 1)
                    {
                        expected_tensor = torch::sigmoid(expected_tensor);
                    }
                    else
                    {
                        expected_tensor = torch::relu(expected_tensor);
                    }
                    if (!l)
                    {
                        error = tensor_matrix_multiplication(input[i][j][k],
                                                             weights[i][j][k][l],
                                                             returned_tensors_i[i][j][k][l]);
                    }
                    else
                    {
                        error = tensor_matrix_multiplication(returned_tensors[i][j][k][l - 1],
                                                             weights[i][j][k][l],
                                                             returned_tensors_i[i][j][k][l]);
                    }
                    ck_assert_ptr_null(error);

                    error = tensor_addition(returned_tensors_i[i][j][k][l],
                                            bias[i][j][k][l],
                                            returned_tensors_j[i][j][k][l]);
                    ck_assert_ptr_null(error);

                    if (l == LAYERS - 1)
                    {
                        error = tensor_sigmoid(returned_tensors_j[i][j][k][l],
                                               returned_tensors[i][j][k][l]);
                    }
                    else
                    {
                        error = tensor_rectified_linear(returned_tensors_j[i][j][k][l],
                                                        returned_tensors[i][j][k][l]);
                    }
                    ck_assert_ptr_null(error);
                    
                    error = view_create(&view,
                                        (uint64_t) expected_tensor.storage_offset(),
                                        (uint64_t) expected_tensor.ndimension(),
                                        (uint64_t *) expected_tensor.sizes().data(),
                                        (uint64_t *) expected_tensor.strides().data());
                    ck_assert_ptr_null(error);

                    error = storage_create(&storage,
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) expected_tensor.storage().nbytes() / 
                                           (uint64_t) datatype_size((datatype_t) j),
                                           (void *) expected_tensor.data_ptr());
                    ck_assert_ptr_null(error);

                    error = buffer_create(&buffer,
                                          view,
                                          storage,
                                          false);
                    ck_assert_ptr_null(error);

                    error = tensor_create(&expected_tensors[i][j][k][l],
                                          buffer,
                                          NULL,
                                          NULL,
                                          expected_tensor.requires_grad(),
                                          false);
                    ck_assert_ptr_null(error);

                    ck_assert_tensor_equiv(returned_tensors[i][j][k][l], 
                                           expected_tensors[i][j][k][l]);
                }
                
                // Backward Propogation
                expected_tensor.mean().backward();

                error = tensor_mean(returned_tensors[i][j][k][LAYERS - 1],
                                    cost[i][j][k],
                                    NULL,
                                    returned_tensors[i][j][k][LAYERS - 1]->buffer->view->rank,
                                    false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost[i][j][k], NULL);
                ck_assert_ptr_null(error);

                for (int l = 0; l < LAYERS; l++)
                {
                    error = view_create(&view,
                                        (uint64_t) torch_weights[i][j][k][l].grad().storage_offset(),
                                        (uint64_t) torch_weights[i][j][k][l].grad().ndimension(),
                                        (uint64_t *) torch_weights[i][j][k][l].grad().sizes().data(),
                                        (uint64_t *) torch_weights[i][j][k][l].grad().strides().data());
                    ck_assert_ptr_null(error);

                    error = storage_create(&storage,
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) torch_weights[i][j][k][l].grad().storage().nbytes() / 
                                           (uint64_t) datatype_size((datatype_t) j),
                                           (void *) torch_weights[i][j][k][l].grad().data_ptr());
                    ck_assert_ptr_null(error);

                    error = buffer_create(&buffer,
                                          view,
                                          storage,
                                          false);
                    ck_assert_ptr_null(error);

                    error = tensor_create(&expected_gradients_weights[i][j][k][l],
                                          buffer,
                                          NULL,
                                          NULL,
                                          false,
                                          false);
                    ck_assert_ptr_null(error);

                    error = view_create(&view,
                                        (uint64_t) torch_bias[i][j][k][l].grad().storage_offset(),
                                        (uint64_t) torch_bias[i][j][k][l].grad().ndimension(),
                                        (uint64_t *) torch_bias[i][j][k][l].grad().sizes().data(),
                                        (uint64_t *) torch_bias[i][j][k][l].grad().strides().data());
                    ck_assert_ptr_null(error);

                    error = storage_create(&storage,
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) torch_bias[i][j][k][l].grad().storage().nbytes() / 
                                           (uint64_t) datatype_size((datatype_t) j),
                                           (void *) torch_bias[i][j][k][l].grad().data_ptr());
                    ck_assert_ptr_null(error);

                    error = buffer_create(&buffer,
                                          view,
                                          storage,
                                          false);
                    ck_assert_ptr_null(error);

                    error = tensor_create(&expected_gradients_bias[i][j][k][l],
                                          buffer,
                                          NULL,
                                          NULL,
                                          false,
                                          false);
                    ck_assert_ptr_null(error);

                    ck_assert_tensor_equiv(weights[i][j][k][l]->gradient,
                                           expected_gradients_weights[i][j][k][l]);
                    ck_assert_tensor_equiv(bias[i][j][k][l]->gradient,
                                           expected_gradients_bias[i][j][k][l]);
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

    sr = srunner_create(make_binary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}