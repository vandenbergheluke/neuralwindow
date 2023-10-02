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

typedef enum tensor_structure_type_t
{
    TENSOR_EXPAND,
    TENSOR_PERMUTE,
    TENSOR_RESHAPE,
    TENSOR_SLICE,
    TENSOR_PADDING,
} tensor_structure_type_t;

#define EXPAND_CASES 1
#define PERMUTE_CASES 1
#define RESHAPE_CASES 1
#define SLICE_CASES 1
#define PADDING_CASES 1
#define OPERATIONS 5

nw_error_t *error = NULL;

std::vector<tensor_t *> tensors[OPERATIONS][RUNTIMES][DATATYPES];
std::vector<tensor_t *> returned_tensors[OPERATIONS][RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_tensors[OPERATIONS][RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradient[OPERATIONS][RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_tensors[OPERATIONS][RUNTIMES][DATATYPES];

std::vector<int64_t> expand_shapes[EXPAND_CASES] = {
    {},
};
std::vector<int64_t> expand_arguments[EXPAND_CASES] = {
    {1},
};

std::vector<int64_t> permute_shapes[PERMUTE_CASES] = {
    {1, 2},
};
std::vector<int64_t> permute_arguments[PERMUTE_CASES] = {
    {1, 0},
};

std::vector<int64_t> reshape_shapes[RESHAPE_CASES] = {
    {2, 3},
};
std::vector<int64_t> reshape_arguments[RESHAPE_CASES] = {
    {6},
};

std::vector<int64_t> slice_shapes[SLICE_CASES] = {
    {2},
};
std::vector<int64_t> slice_arguments[SLICE_CASES] = {
    {1, 2},
};

std::vector<int64_t> padding_shapes[PADDING_CASES] = {
    {1},
};
std::vector<int64_t> padding_arguments[PADDING_CASES] = {
    {1, 1},
};

void setup(tensor_structure_type_t tensor_structure_type, std::vector<int64_t> *shapes, int cases)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < cases; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors[tensor_structure_type][i][j].push_back(torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)));
                    break;
                case FLOAT64:
                    torch_tensors[tensor_structure_type][i][j].push_back(torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors[tensor_structure_type][i][j][k].retain_grad();
                tensors[tensor_structure_type][i][j].push_back(torch_to_tensor(torch_tensors[tensor_structure_type][i][j][k], (runtime_t) i, (datatype_t) j));
                returned_tensors[tensor_structure_type][i][j].push_back(NULL);
            }
        }
    }
}

void teardown(tensor_structure_type_t tensor_structure_type, int cases)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < cases; k++)
            {
                tensor_destroy(tensors[tensor_structure_type][i][j][k]);
                // tensor_destroy(expected_tensors[tensor_structure_type][i][j][k]);
                // tensor_destroy(expected_gradient[tensor_structure_type][i][j][k]);
            }
            tensors[tensor_structure_type][i][j].clear();
            expected_tensors[tensor_structure_type][i][j].clear();
    //         // expected_gradient[tensor_structure_type][i][j].clear();
        }
    }
    error_print(error);
    error_destroy(error);
}

void test_structure(tensor_structure_type_t tensor_structure_type, std::vector<int64_t> *arguments, int cases)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < cases; k++)
            {
                torch::Tensor expected_tensor;

                void *original = NULL;
                switch (tensor_structure_type)
                {
                case TENSOR_EXPAND:
                    expected_tensor = torch_tensors[tensor_structure_type][i][j][k].expand(arguments[k]);
                    break;
                case TENSOR_PERMUTE:
                    expected_tensor = torch_tensors[tensor_structure_type][i][j][k].permute(arguments[k]);
                    break;
                case TENSOR_RESHAPE:
                    expected_tensor = torch_tensors[tensor_structure_type][i][j][k].reshape(arguments[k]);
                    break;
                case TENSOR_SLICE:
                    expected_tensor = torch_tensors[tensor_structure_type][i][j][k];
                    original = expected_tensor.storage().data_ptr().get();
                    for (int64_t l = 0; l < expected_tensor.ndimension(); l++)
                    {
                        expected_tensor = expected_tensor.slice(l, arguments[k][2 * l], arguments[k][2 * l + 1]);
                    }
                    printf("%s\n", (expected_tensor.storage().data_ptr().get() == original) ? "true" : "false");
                    break;
                case TENSOR_PADDING:
                    expected_tensor = torch::nn::functional::pad(torch_tensors[tensor_structure_type][i][j][k], torch::nn::functional::PadFuncOptions(arguments[k]));
                    break;
                default:
                    ck_abort_msg("unknown structure type.");
                }

                expected_tensors[tensor_structure_type][i][j].push_back(torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j));

                switch (tensor_structure_type)
                {
                case TENSOR_EXPAND:
                    error = tensor_expand(tensors[tensor_structure_type][i][j][k], 
                                          (uint64_t *) arguments[k].data(),
                                          (uint64_t) arguments[k].size(),
                                          &returned_tensors[tensor_structure_type][i][j][k]);
                    break;
                case TENSOR_PERMUTE:
                    error = tensor_permute(tensors[tensor_structure_type][i][j][k],
                                           &returned_tensors[tensor_structure_type][i][j][k],
                                           (uint64_t *) arguments[k].data(),
                                           (uint64_t) arguments[k].size());
                    break;
                case TENSOR_RESHAPE:
                    error = tensor_reshape(tensors[tensor_structure_type][i][j][k],
                                           &returned_tensors[tensor_structure_type][i][j][k],
                                           (uint64_t *) arguments[k].data(),
                                           (uint64_t) arguments[k].size());
                    break;
                case TENSOR_SLICE:
                    error = tensor_slice(tensors[tensor_structure_type][i][j][k],
                                         &returned_tensors[tensor_structure_type][i][j][k],
                                         (uint64_t *) arguments[k].data(),
                                         (uint64_t) arguments[k].size());
                    break;
                case TENSOR_PADDING:
                    error = tensor_padding(tensors[tensor_structure_type][i][j][k],
                                           &returned_tensors[tensor_structure_type][i][j][k],
                                           (uint64_t *) arguments[k].data(),
                                           (uint64_t) arguments[k].size());
                    break;
                default:
                    ck_abort_msg("unknown structure type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(returned_tensors[tensor_structure_type][i][j][k], expected_tensors[tensor_structure_type][i][j][k]);

                // Back prop
                // expected_tensor.sum().backward();
                // expected_gradient[tensor_structure_type][i][j].push_back(torch_to_tensor(torch_tensors[tensor_structure_type][i][j][k].grad(), (runtime_t) i, (datatype_t) j));
                // tensor_t *cost = NULL;
                // error = tensor_summation(returned_tensors[tensor_structure_type][i][j][k], &cost, NULL, 1, false);
                // ck_assert_ptr_null(error);
                // error = tensor_backward(cost, NULL);
                // ck_assert_ptr_null(error);

                // ck_assert_tensor_equiv(tensors[tensor_structure_type][i][j][k]->gradient, expected_gradient[tensor_structure_type][i][j][k]);
            }
        }
    }
}

void setup_expand(void)
{
    setup(TENSOR_EXPAND, expand_shapes, EXPAND_CASES);
}

void teardown_expand(void)
{
    teardown(TENSOR_EXPAND, EXPAND_CASES);
}

START_TEST(test_expand)
{
   test_structure(TENSOR_EXPAND, expand_arguments, EXPAND_CASES);
}
END_TEST

void setup_permute(void)
{
    setup(TENSOR_PERMUTE, permute_shapes, PERMUTE_CASES);
}

void teardown_permute(void)
{
    teardown(TENSOR_PERMUTE, PERMUTE_CASES);
}

START_TEST(test_permute)
{
   test_structure(TENSOR_PERMUTE, permute_arguments, PERMUTE_CASES);
}
END_TEST

void setup_reshape(void)
{
    setup(TENSOR_RESHAPE, reshape_shapes, RESHAPE_CASES);
}

void teardown_reshape(void)
{
    teardown(TENSOR_RESHAPE, RESHAPE_CASES);
}

START_TEST(test_reshape)
{
   test_structure(TENSOR_RESHAPE, reshape_arguments, RESHAPE_CASES);
}
END_TEST

void setup_slice(void)
{
    setup(TENSOR_SLICE, slice_shapes, SLICE_CASES);
}

void teardown_slice(void)
{
    teardown(TENSOR_SLICE, SLICE_CASES);
}

START_TEST(test_slice)
{
   test_structure(TENSOR_SLICE, slice_arguments, SLICE_CASES);
}
END_TEST

void setup_padding(void)
{
    setup(TENSOR_PADDING, padding_shapes, PADDING_CASES);
}

void teardown_padding(void)
{
    teardown(TENSOR_PADDING, PADDING_CASES);
}

START_TEST(test_padding)
{
   test_structure(TENSOR_PADDING, padding_arguments, PADDING_CASES);
}
END_TEST

Suite *make_structure_suite(void)
{
    Suite *s;
    TCase *tc_expand;
    TCase *tc_permute;
    TCase *tc_reshape;
    TCase *tc_slice;
    TCase *tc_padding;

    s = suite_create("Test structure Tensor Suite");
    // tc_expand = tcase_create("Test Expand Tensor Case");
    // tc_permute = tcase_create("Test Permute Tensor Case");
    // tc_reshape = tcase_create("Test Reshape Tensor Case");
    tc_slice = tcase_create("Test Slice Tensor Case");
    // tc_padding = tcase_create("Test Padding Tensor Case");

    // tcase_add_checked_fixture(tc_expand, setup_expand, teardown_expand);
    // tcase_add_checked_fixture(tc_permute, setup_permute, teardown_permute);
    // tcase_add_checked_fixture(tc_reshape, setup_reshape, teardown_reshape);
    tcase_add_checked_fixture(tc_slice, setup_slice, teardown_slice);
    // tcase_add_checked_fixture(tc_padding, setup_padding, teardown_padding);

    // tcase_add_test(tc_expand, test_expand);
    // tcase_add_test(tc_permute, test_permute);
    // tcase_add_test(tc_reshape, test_reshape);
    tcase_add_test(tc_slice, test_slice);
    // tcase_add_test(tc_padding, test_padding);

    // suite_add_tcase(s, tc_expand);
    // suite_add_tcase(s, tc_permute);
    // suite_add_tcase(s, tc_reshape);
    suite_add_tcase(s, tc_slice);
    // suite_add_tcase(s, tc_padding);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_structure_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}