#include <iostream>
extern "C"
{
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <graph.h>
}
#include <test_helper.h>
#include <torch/torch.h>

#define CASES 6

tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype);

nw_error_t *error;

tensor_t *tensors_x[CASES];
tensor_t *tensors_y[CASES];
tensor_t *tensors_x_output[CASES];
tensor_t *tensors_y_output[CASES];
tensor_t *returned_tensors[CASES];

torch::Tensor torch_tensors_x[CASES];
torch::Tensor torch_tensors_y[CASES];

std::vector<int64_t> shapes[CASES] = {
    {6, 5, 4, 3, 2},
    {6, 5, 4, 2, 1},
};

void setup(void)
{
    for (int i = 0; i < CASES; ++i)
    {
            tensors_x[i] = NULL;
            tensors_x_output[i] = NULL;
            tensors_y[i] = NULL;
            tensors_y_output[i] = NULL;
            returned_tensors[i] = NULL;

            torch_tensors_x[i] = torch::randn(shapes[0], torch::TensorOptions().dtype(torch::kFloat64));
            torch_tensors_y[i] = torch::randn(shapes[0], torch::TensorOptions().dtype(torch::kFloat64));

            tensors_x[i] = torch_to_tensor(torch_tensors_x[i], MKL_RUNTIME, FLOAT64);
            tensors_x_output[i] = torch_to_tensor(torch_tensors_x[i], MKL_RUNTIME, FLOAT64);
            tensors_y[i] = torch_to_tensor(torch_tensors_y[i], MKL_RUNTIME, FLOAT64);
            tensors_y_output[i] = torch_to_tensor(torch_tensors_y[i], MKL_RUNTIME, FLOAT64);
            returned_tensors[i] = torch_to_tensor(torch_tensors_x[i], MKL_RUNTIME, FLOAT64);
    }
}

void teardown(void)
{
    for (int i = 0; i < CASES; ++i)
    {

        tensor_destroy(tensors_x[i]);
        tensor_destroy(tensors_y[i]);
        tensor_destroy(tensors_x_output[i]);
        tensor_destroy(tensors_y_output[i]);
        tensor_destroy(returned_tensors[i]);
    }
    error_print(error);
    error_destroy(error);
    destroy_graph();
}

void test_graph(void)
{
    //logarith on x
    for (int i = 0; i < CASES; ++i)
    {
        error = tensor_rectified_linear(tensors_x[i], &tensors_x_output[i]);
        ck_assert_ptr_null(error);

        error = tensor_exponential(tensors_y[i], &tensors_y_output[i]);
        ck_assert_ptr_null(error);

        error = tensor_addition(tensors_x_output[i], tensors_y_output[i], &returned_tensors[i]);
        ck_assert_ptr_null(error);
    }
}

START_TEST(test_null_errors)
{
    test_graph();
}
END_TEST

Suite *make_graph_suite(void)
{
    Suite *s;
    TCase *tc_graph;

    s = suite_create("Test Graph Suite");

    tc_graph = tcase_create("Graph Case");
    tcase_add_checked_fixture(tc_graph, setup, teardown);
    tcase_add_test(tc_graph, test_null_errors);

    suite_add_tcase(s, tc_graph);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_graph_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE; 
}