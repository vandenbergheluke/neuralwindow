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

#define CONVOLUTION_CASES 3
#define CONVOLUTION_TRANSPOSE_CASES 3

std::vector<int64_t> convolution_2d_shapes_w[CONVOLUTION_CASES] = {
    {5, 3, 6, 7},
    {1, 1, 9, 7},
    {10, 9, 9, 11},
};

std::vector<int64_t> convolution_2d_shapes_x[CONVOLUTION_CASES] = {
    {5, 3, 3, 3},
    {10, 1, 4, 4},
    {11, 9, 1, 1},
};

std::vector<int64_t> convolution_2d_shapes_y[CONVOLUTION_CASES] = {
    {5},
    {10},
    {11},
};

int64_t convolution_2d_stride[CONVOLUTION_CASES] = {
    2,
    1,
    3,
};

int64_t convolution_2d_padding[CONVOLUTION_CASES] = {
    1,
    3,
    0,
};

std::vector<int64_t> convolution_2d_transpose_shapes_w[CONVOLUTION_TRANSPOSE_CASES] = {
    {5, 3, 6, 7},
    {10, 4, 7, 7},
    {2, 5, 3, 8},
};

std::vector<int64_t> convolution_2d_transpose_shapes_x[CONVOLUTION_TRANSPOSE_CASES] = {
    {3, 2, 3, 3},
    {4, 3, 2, 2},
    {5, 1, 5, 5},
};

std::vector<int64_t> convolution_2d_transpose_shapes_y[CONVOLUTION_TRANSPOSE_CASES] = {
    {2},
    {3},
    {1},
};

int64_t convolution_2d_transpose_stride[CONVOLUTION_TRANSPOSE_CASES] = {
    2,
    2,
    1,
};

int64_t convolution_2d_transpose_padding[CONVOLUTION_TRANSPOSE_CASES] = {
    1,
    2,
    3,
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
    TENSOR_CONVOLUTION_TRANSPOSE,
} tensor_reduction_operation_type_t;

int cases(tensor_ternary_operation_type_t tensor_ternary_operation_type)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION:
        return CONVOLUTION_CASES;
    case TENSOR_CONVOLUTION_TRANSPOSE:
        return CONVOLUTION_TRANSPOSE_CASES;
    default:
        return 0;
    }
}

std::vector<int64_t> shapes_w(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION:
        return convolution_2d_shapes_w[i];
    case TENSOR_CONVOLUTION_TRANSPOSE:
        return convolution_2d_transpose_shapes_w[i];
    default:
        return std::vector<int64_t>{};
    }
}

std::vector<int64_t> shapes_x(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION:
        return convolution_2d_shapes_x[i];
    case TENSOR_CONVOLUTION_TRANSPOSE:
        return convolution_2d_transpose_shapes_x[i];
    default:
        return std::vector<int64_t>{};
    }
}

std::vector<int64_t> shapes_y(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION:
        return convolution_2d_shapes_y[i];
    case TENSOR_CONVOLUTION_TRANSPOSE:
        return convolution_2d_transpose_shapes_y[i];
    default:
        return std::vector<int64_t>{};
    }
}

int64_t stride(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION:
        return convolution_2d_stride[i];
    case TENSOR_CONVOLUTION_TRANSPOSE:
        return convolution_2d_transpose_stride[i];
    default:
        return 0;
    }
}

int64_t padding(tensor_ternary_operation_type_t tensor_ternary_operation_type, int i)
{
    switch(tensor_ternary_operation_type)
    {
    case TENSOR_CONVOLUTION:
        return convolution_2d_padding[i];
    case TENSOR_CONVOLUTION_TRANSPOSE:
        return convolution_2d_transpose_padding[i];
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
                    torch_tensors_w[i][j][k] = torch::randn(shapes_w(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    break;
                case FLOAT64:
                    torch_tensors_w[i][j][k] = torch::randn(shapes_w(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x(tensor_ternary_operation_type, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat64)
                                                            .requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y(tensor_ternary_operation_type, k),
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
    const int CASES = cases(tensor_ternary_operation_type);
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
                                                                    torch::nn::functional::Conv2dFuncOptions().padding(padding(tensor_ternary_operation_type, k))
                                                                                                              .stride(stride(tensor_ternary_operation_type, k))
                                                                                                              .bias(torch_tensors_y[i][j][k]));
                    break;
                case TENSOR_CONVOLUTION_TRANSPOSE:
                    expected_tensor = torch::nn::functional::conv_transpose2d(torch_tensors_w[i][j][k], 
                                                                            torch_tensors_x[i][j][k], 
                                                                            torch::nn::functional::ConvTranspose2dFuncOptions().padding(padding(tensor_ternary_operation_type, k))
                                                                                                                                .stride(stride(tensor_ternary_operation_type, k))
                                                                                                                                .bias(torch_tensors_y[i][j][k]));
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (tensor_ternary_operation_type)
                {
                case TENSOR_CONVOLUTION:
                    error = tensor_convolution_2d(tensors_w[i][j][k], tensors_x[i][j][k], tensors_y[i][j][k], 
                                               &returned_tensors[i][j][k], stride(tensor_ternary_operation_type, k),
                                               padding(tensor_ternary_operation_type, k));
                    break;
                case TENSOR_CONVOLUTION_TRANSPOSE:
                    error = tensor_convolution_2d_transpose(tensors_w[i][j][k], tensors_x[i][j][k], tensors_y[i][j][k], 
                                                         &returned_tensors[i][j][k], stride(tensor_ternary_operation_type, k), 
                                                         padding(tensor_ternary_operation_type, k));
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

void setup_convolution_2d(void)
{
    setup(TENSOR_CONVOLUTION);
}

void setup_convolution_2d_transpose(void)
{
    setup(TENSOR_CONVOLUTION_TRANSPOSE);
}

void teardown_convolution_2d(void)
{
    teardown(TENSOR_CONVOLUTION);
}

void teardown_convolution_2d_transpose(void)
{
    teardown(TENSOR_CONVOLUTION_TRANSPOSE);
}

START_TEST(test_tensor_convolution_2d)
{
    test_ternary(TENSOR_CONVOLUTION);
}
END_TEST

START_TEST(test_tensor_convolution_2d_transpose)
{
    test_ternary(TENSOR_CONVOLUTION_TRANSPOSE);
}
END_TEST

Suite *make_ternary_suite(void)
{
    Suite *s;
    TCase *tc_convolution_2d;
    TCase *tc_convolution_2d_transpose;

    s = suite_create("Test Ternary Tensor Suite");

    tc_convolution_2d = tcase_create("Test Convolution Case");
    tcase_add_checked_fixture(tc_convolution_2d, setup_convolution_2d, teardown_convolution_2d);
    tcase_add_test(tc_convolution_2d, test_tensor_convolution_2d);

    tc_convolution_2d_transpose = tcase_create("Test Convolution Transpose Case");
    tcase_add_checked_fixture(tc_convolution_2d_transpose, setup_convolution_2d_transpose, teardown_convolution_2d_transpose);
    tcase_add_test(tc_convolution_2d_transpose, test_tensor_convolution_2d_transpose);

    suite_add_tcase(s, tc_convolution_2d);
    suite_add_tcase(s, tc_convolution_2d_transpose);

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