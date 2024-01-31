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

#define CONVOLUTION_2D_CASES 3
#define CONVOLUTION_TRANSPOSE_2D_CASES 3
#define BATCH_NORMALIZATION_2D_CASES 3

std::vector<int64_t> convolution_2d_shapes_x[CONVOLUTION_2D_CASES] = {
    {5, 3, 6, 7},
    {1, 1, 9, 7},
    {10, 9, 9, 11},
};

std::vector<int64_t> convolution_2d_shapes_weights[CONVOLUTION_2D_CASES] = {
    {5, 3, 3, 3},
    {10, 1, 4, 4},
    {11, 9, 1, 1},
};

std::vector<int64_t> convolution_2d_shapes_bias[CONVOLUTION_2D_CASES] = {
    {5},
    {10},
    {11},
};

int64_t convolution_2d_stride[CONVOLUTION_2D_CASES] = {
    2,
    1,
    3,
};

int64_t convolution_2d_padding[CONVOLUTION_2D_CASES] = {
    1,
    3,
    0,
};

std::vector<int64_t> convolution_transpose_2d_shapes_x[CONVOLUTION_TRANSPOSE_2D_CASES] = {
    {5, 3, 6, 7},
    {10, 4, 7, 7},
    {2, 5, 3, 8},
};

std::vector<int64_t> convolution_transpose_2d_shapes_weights[CONVOLUTION_TRANSPOSE_2D_CASES] = {
    {3, 2, 3, 3},
    {4, 3, 2, 2},
    {5, 1, 5, 5},
};

std::vector<int64_t> convolution_transpose_2d_shapes_bias[CONVOLUTION_TRANSPOSE_2D_CASES] = {
    {2},
    {3},
    {1},
};

int64_t convolution_transpose_2d_stride[CONVOLUTION_TRANSPOSE_2D_CASES] = {
    2,
    2,
    1,
};

int64_t convolution_transpose_2d_padding[CONVOLUTION_TRANSPOSE_2D_CASES] = {
    1,
    2,
    3,
};

std::vector<int64_t> batch_normalization_2d_shapes_x[BATCH_NORMALIZATION_2D_CASES] = {
    {3, 2, 3, 3},
    {7, 3, 5, 4},
    {7, 1, 2, 3},
};

std::vector<int64_t> batch_normalization_2d_features[BATCH_NORMALIZATION_2D_CASES] = {
    {2},
    {3},
    {1},
};

float32_t batch_normalization_2d_momentum_f[BATCH_NORMALIZATION_2D_CASES] = {
    0.2,
    0.1,
    0.3,
};

float64_t batch_normalization_2d_momentum[BATCH_NORMALIZATION_2D_CASES] = {
    0.2,
    0.1,
    0.3,
};

float32_t batch_normalization_2d_epsilon_f[BATCH_NORMALIZATION_2D_CASES] = {
    1e-6,
    1e-4,
    1e-7,
};

float64_t batch_normalization_2d_epsilon[BATCH_NORMALIZATION_2D_CASES] = {
    1e-6,
    1e-4,
    1e-7,
};

bool_t batch_normalization_2d_inference[BATCH_NORMALIZATION_2D_CASES] = {
    false,
    true,
    false,
};

nw_error_t *error = NULL;

// All
std::vector<tensor_t *> tensors_x[RUNTIMES][DATATYPES];
std::vector<tensor_t *> tensors_weights[RUNTIMES][DATATYPES];
std::vector<tensor_t *> tensors_bias[RUNTIMES][DATATYPES];
std::vector<tensor_t *> returned_tensors[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_tensors[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_x[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_weights[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_bias[RUNTIMES][DATATYPES];

std::vector<torch::Tensor> torch_tensors_x[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_tensors_weights[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_tensors_bias[RUNTIMES][DATATYPES];

// Batch normalization specific
std::vector<tensor_t *> running_means[RUNTIMES][DATATYPES];
std::vector<tensor_t *> running_variances[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_running_means[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_running_variances[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_running_means[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_running_variances[RUNTIMES][DATATYPES];

typedef enum tensor_ternary_operation_type_t
{
    TENSOR_CONVOLUTION_2D,
    TENSOR_CONVOLUTION_TRANSPOSE_2D,
    TENSOR_BATCH_NORMALIZATION_2D,
} tensor_reduction_operation_type_t;

int cases(tensor_ternary_operation_type_t tensor_ternary_operation_type)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION_2D:
        return CONVOLUTION_2D_CASES;
    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
        return CONVOLUTION_TRANSPOSE_2D_CASES;
    case TENSOR_BATCH_NORMALIZATION_2D:
        return BATCH_NORMALIZATION_2D_CASES;
    default:
        return 0;
    }
}

std::vector<int64_t> shapes_x(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION_2D:
        return convolution_2d_shapes_x[i];
    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
        return convolution_transpose_2d_shapes_x[i];
    case TENSOR_BATCH_NORMALIZATION_2D:
        return batch_normalization_2d_shapes_x[i];
    default:
        return std::vector<int64_t>{};
    }
}

std::vector<int64_t> shapes_weights(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION_2D:
        return convolution_2d_shapes_weights[i];
    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
        return convolution_transpose_2d_shapes_weights[i];
    case TENSOR_BATCH_NORMALIZATION_2D:
        return batch_normalization_2d_features[i];
    default:
        return std::vector<int64_t>{};
    }
}

std::vector<int64_t> shapes_bias(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION_2D:
        return convolution_2d_shapes_bias[i];
    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
        return convolution_transpose_2d_shapes_bias[i];
    case TENSOR_BATCH_NORMALIZATION_2D:
        return batch_normalization_2d_features[i];
    default:
        return std::vector<int64_t>{};
    }
}

int64_t stride(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION_2D:
        return convolution_2d_stride[i];
    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
        return convolution_transpose_2d_stride[i];
    default:
        return 0;
    }
}

int64_t padding(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION_2D:
        return convolution_2d_padding[i];
    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
        return convolution_transpose_2d_padding[i];
    default:
        return 0;
    }
}

void setup(tensor_ternary_operation_type_t tensor_ternary_operation_type)
{
    const int CASES = cases(tensor_ternary_operation_type);
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            tensors_x[i][j] = std::vector<tensor_t *>(CASES);
            tensors_weights[i][j] = std::vector<tensor_t *>(CASES);
            tensors_bias[i][j] = std::vector<tensor_t *>(CASES);
            returned_tensors[i][j] = std::vector<tensor_t *>(CASES);
            expected_tensors[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_x[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_weights[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_bias[i][j] = std::vector<tensor_t *>(CASES);
            torch_tensors_x[i][j] = std::vector<torch::Tensor>(CASES);
            torch_tensors_weights[i][j] = std::vector<torch::Tensor>(CASES);
            torch_tensors_bias[i][j] = std::vector<torch::Tensor>(CASES);
            switch (tensor_ternary_operation_type)
            {
            case TENSOR_CONVOLUTION_2D:
            case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                break;
            case TENSOR_BATCH_NORMALIZATION_2D:
                running_means[i][j] = std::vector<tensor_t *>(CASES);
                running_variances[i][j] = std::vector<tensor_t *>(CASES);
                expected_running_means[i][j] = std::vector<tensor_t *>(CASES);
                expected_running_variances[i][j] = std::vector<tensor_t *>(CASES);
                torch_running_means[i][j] = std::vector<torch::Tensor>(CASES); 
                torch_running_variances[i][j] = std::vector<torch::Tensor>(CASES); 
                break;
            default:
                ck_abort_msg("unknown operation type.");
            }

            for (int k = 0; k < CASES; ++k)
            {
                tensors_x[i][j][k] = NULL;
                tensors_weights[i][j][k] = NULL;
                tensors_bias[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradients_x[i][j][k] = NULL;
                expected_gradients_weights[i][j][k] = NULL;
                expected_gradients_bias[i][j][k] = NULL;
                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION_2D:
                case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                    break;
                case TENSOR_BATCH_NORMALIZATION_2D:
                    running_means[i][j][k] = NULL;
                    running_variances[i][j][k] = NULL;
                    expected_running_means[i][j][k] = NULL;
                    expected_running_variances[i][j][k] = NULL;
                    break;
                default:
                    ck_abort_msg("unknown operation type.");
                }

                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_weights[i][j][k] = torch::randn(shapes_weights(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_bias[i][j][k] = torch::randn(shapes_bias(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    switch (tensor_ternary_operation_type)
                    {
                    case TENSOR_CONVOLUTION_2D:
                    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                        break;
                    case TENSOR_BATCH_NORMALIZATION_2D:
                        torch_running_means[i][j][k] = torch::randn(batch_normalization_2d_features[k], torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
                        torch_running_variances[i][j][k] = torch::rand(batch_normalization_2d_features[k], torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
                        break;
                    default:
                        ck_abort_msg("unknown operation type.");
                    }
                    break;
                case FLOAT64:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    torch_tensors_weights[i][j][k] = torch::randn(shapes_weights(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    torch_tensors_bias[i][j][k] = torch::randn(shapes_bias(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    switch (tensor_ternary_operation_type)
                    {
                    case TENSOR_CONVOLUTION_2D:
                    case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                        break;
                    case TENSOR_BATCH_NORMALIZATION_2D:
                        torch_running_means[i][j][k] = torch::randn(batch_normalization_2d_features[k], torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
                        torch_running_variances[i][j][k] = torch::rand(batch_normalization_2d_features[k], torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
                        break;
                    default:
                        ck_abort_msg("unknown operation type.");
                    }
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors_x[i][j][k].retain_grad();
                torch_tensors_weights[i][j][k].retain_grad();
                torch_tensors_bias[i][j][k].retain_grad();

                tensors_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k], (runtime_t) i, (datatype_t) j);
                tensors_weights[i][j][k] = torch_to_tensor(torch_tensors_weights[i][j][k], (runtime_t) i, (datatype_t) j);
                tensors_bias[i][j][k] = torch_to_tensor(torch_tensors_bias[i][j][k], (runtime_t) i, (datatype_t) j);
                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION_2D:
                case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                    break;
                case TENSOR_BATCH_NORMALIZATION_2D:
                    running_means[i][j][k] = torch_to_tensor(torch_running_means[i][j][k], (runtime_t) i, (datatype_t) j);
                    running_variances[i][j][k] = torch_to_tensor(torch_running_variances[i][j][k], (runtime_t) i, (datatype_t) j);
                    break;
                default:
                    ck_abort_msg("unknown operation type.");
                }
            }
        }
    }
}

void teardown(tensor_ternary_operation_type_t tensor_ternary_operation_type)
{
    const int CASES = cases(tensor_ternary_operation_type);
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                tensor_destroy(tensors_x[i][j][k]);
                tensor_destroy(tensors_weights[i][j][k]);
                tensor_destroy(tensors_bias[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
                tensor_destroy(expected_gradients_x[i][j][k]);
                tensor_destroy(expected_gradients_weights[i][j][k]);
                tensor_destroy(expected_gradients_bias[i][j][k]);
                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION_2D:
                case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                    break;
                case TENSOR_BATCH_NORMALIZATION_2D:
                    tensor_destroy(running_means[i][j][k]);
                    tensor_destroy(running_variances[i][j][k]);
                    tensor_destroy(expected_running_means[i][j][k]);
                    tensor_destroy(expected_running_variances[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unknown operation type.");
                }
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_ternary(tensor_ternary_operation_type_t tensor_ternary_operation_type)
{
    const int CASES = cases(tensor_ternary_operation_type);
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            datatype_t datatype = (datatype_t) j;
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;

                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION_2D:
                    expected_tensor = torch::nn::functional::conv2d(torch_tensors_x[i][j][k], 
                                                                    torch_tensors_weights[i][j][k], 
                                                                    torch::nn::functional::Conv2dFuncOptions().padding(padding(tensor_ternary_operation_type, k))
                                                                                                              .stride(stride(tensor_ternary_operation_type, k))
                                                                                                              .bias(torch_tensors_bias[i][j][k]));
                    break;
                case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                    expected_tensor = torch::nn::functional::conv_transpose2d(torch_tensors_x[i][j][k], 
                                                                            torch_tensors_weights[i][j][k], 
                                                                            torch::nn::functional::ConvTranspose2dFuncOptions().padding(padding(tensor_ternary_operation_type, k))
                                                                                                                                .stride(stride(tensor_ternary_operation_type, k))
                                                                                                                                .bias(torch_tensors_bias[i][j][k]));
                    break;
                case TENSOR_BATCH_NORMALIZATION_2D:
                    switch (datatype)
                    {
                    case FLOAT32:
                        expected_tensor = torch::nn::functional::batch_norm(torch_tensors_x[i][j][k], torch_running_means[i][j][k], torch_running_variances[i][j][k], 
                                                                            torch::nn::functional::BatchNormFuncOptions().eps(batch_normalization_2d_epsilon_f[k])
                                                                            .momentum(batch_normalization_2d_momentum_f[k]).training(!batch_normalization_2d_inference[k])
                                                                            .weight(torch_tensors_weights[i][j][k]).bias(torch_tensors_bias[i][j][k]));
                        break;
                    case FLOAT64:
                        expected_tensor = torch::nn::functional::batch_norm(torch_tensors_x[i][j][k], torch_running_means[i][j][k], torch_running_variances[i][j][k], 
                                                                            torch::nn::functional::BatchNormFuncOptions().eps(batch_normalization_2d_epsilon[k])
                                                                            .momentum(batch_normalization_2d_momentum[k]).training(!batch_normalization_2d_inference[k])
                                                                            .weight(torch_tensors_weights[i][j][k]).bias(torch_tensors_bias[i][j][k]));
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION_2D:
                    error = tensor_convolution_2d(tensors_x[i][j][k], tensors_weights[i][j][k], tensors_bias[i][j][k], 
                                               &returned_tensors[i][j][k], stride(tensor_ternary_operation_type, k),
                                               padding(tensor_ternary_operation_type, k));
                    break;
                case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                    error = tensor_convolution_transpose_2d(tensors_x[i][j][k], tensors_weights[i][j][k], tensors_bias[i][j][k], 
                                                         &returned_tensors[i][j][k], stride(tensor_ternary_operation_type, k), 
                                                         padding(tensor_ternary_operation_type, k));
                    break;
                case TENSOR_BATCH_NORMALIZATION_2D:
                    switch (datatype)
                    {
                    case FLOAT32:
                        error = tensor_batch_normalization_2d(tensors_x[i][j][k], tensors_weights[i][j][k], tensors_bias[i][j][k],
                                                              running_means[i][j][k], running_variances[i][j][k], &returned_tensors[i][j][k],
                                                              batch_normalization_2d_inference[k], &batch_normalization_2d_momentum_f[k],
                                                              &batch_normalization_2d_epsilon_f[k]);
                        break;
                    case FLOAT64:
                        error = tensor_batch_normalization_2d(tensors_x[i][j][k], tensors_weights[i][j][k], tensors_bias[i][j][k],
                                                              running_means[i][j][k], running_variances[i][j][k], &returned_tensors[i][j][k],
                                                              batch_normalization_2d_inference[k], &batch_normalization_2d_momentum[k],
                                                              &batch_normalization_2d_epsilon[k]);
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }
                    break;
                default:
                    ck_abort_msg("unsupported operation type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(returned_tensors[i][j][k], expected_tensors[i][j][k]);
                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION_2D:
                case TENSOR_CONVOLUTION_TRANSPOSE_2D:
                    break;
                case TENSOR_BATCH_NORMALIZATION_2D:
                    expected_running_means[i][j][k] = torch_to_tensor(torch_running_means[i][j][k], (runtime_t) i, (datatype_t) j);
                    expected_running_variances[i][j][k] = torch_to_tensor(torch_running_variances[i][j][k], (runtime_t) i, (datatype_t) j);
                    ck_assert_tensor_equiv(running_means[i][j][k], expected_running_means[i][j][k]);
                    ck_assert_tensor_equiv(running_variances[i][j][k], expected_running_variances[i][j][k]);
                    if (batch_normalization_2d_inference[k])
                    {
                        tensor_destroy(returned_tensors[i][j][k]);
                        continue;
                    }
                    break;
                default:
                    ck_abort_msg("unsupported operation type.");
                }

                expected_tensor.sum().backward();
                expected_gradients_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k].grad(), (runtime_t) i, (datatype_t) j);
                expected_gradients_weights[i][j][k] = torch_to_tensor(torch_tensors_weights[i][j][k].grad(), (runtime_t) i, (datatype_t) j);
                expected_gradients_bias[i][j][k] = torch_to_tensor(torch_tensors_bias[i][j][k].grad(), (runtime_t) i, (datatype_t) j);

                tensor_t *cost = NULL;
                error = tensor_summation(returned_tensors[i][j][k], &cost, NULL, 0, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors_x[i][j][k]->gradient, expected_gradients_x[i][j][k]);
                ck_assert_tensor_equiv(tensors_weights[i][j][k]->gradient, expected_gradients_weights[i][j][k]);
                ck_assert_tensor_equiv(tensors_bias[i][j][k]->gradient, expected_gradients_bias[i][j][k]);
            }
        }
    }
}

void setup_convolution_2d(void)
{
    setup(TENSOR_CONVOLUTION_2D);
}

void teardown_convolution_2d(void)
{
    teardown(TENSOR_CONVOLUTION_2D);
}

void setup_convolution_transpose_2d(void)
{
    setup(TENSOR_CONVOLUTION_TRANSPOSE_2D);
}

void teardown_convolution_transpose_2d(void)
{
    teardown(TENSOR_CONVOLUTION_TRANSPOSE_2D);
}

void setup_batch_normalization_2d(void)
{
    setup(TENSOR_BATCH_NORMALIZATION_2D);
}

void teardown_batch_normalization_2d(void)
{
    teardown(TENSOR_BATCH_NORMALIZATION_2D);
}

START_TEST(test_tensor_convolution_2d)
{
    test_ternary(TENSOR_CONVOLUTION_2D);
}
END_TEST

START_TEST(test_tensor_convolution_transpose_2d)
{
    test_ternary(TENSOR_CONVOLUTION_TRANSPOSE_2D);
}
END_TEST

START_TEST(test_tensor_batch_normalization_2d)
{
    test_ternary(TENSOR_BATCH_NORMALIZATION_2D);
}
END_TEST

Suite *make_ternary_suite(void)
{
    Suite *s;
    TCase *tc_convolution_2d;
    TCase *tc_convolution_transpose_2d;
    TCase *tc_batch_normalization_2d;

    s = suite_create("Test Ternary Tensor Suite");

    tc_convolution_2d = tcase_create("Test Convolution Case");
    tcase_add_checked_fixture(tc_convolution_2d, setup_convolution_2d, teardown_convolution_2d);
    tcase_add_test(tc_convolution_2d, test_tensor_convolution_2d);

    tc_convolution_transpose_2d = tcase_create("Test Convolution Transpose Case");
    tcase_add_checked_fixture(tc_convolution_transpose_2d, setup_convolution_transpose_2d, teardown_convolution_transpose_2d);
    tcase_add_test(tc_convolution_transpose_2d, test_tensor_convolution_transpose_2d);

    tc_batch_normalization_2d = tcase_create("Test Batch Normalization Case");
    tcase_add_checked_fixture(tc_batch_normalization_2d, setup_batch_normalization_2d, teardown_batch_normalization_2d);
    tcase_add_test(tc_batch_normalization_2d, test_tensor_batch_normalization_2d);

    suite_add_tcase(s, tc_convolution_2d);
    suite_add_tcase(s, tc_convolution_transpose_2d);
    suite_add_tcase(s, tc_batch_normalization_2d);

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