#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <tensor.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
}
#include <test_helper.h>

#define CASES 4

nw_error_t *error;

tensor_t *tensors_x[RUNTIMES][DATATYPES][CASES];
tensor_t *tensors_y[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors_x[RUNTIMES][DATATYPES][CASES];
torch::Tensor torch_tensors_y[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    {2, 2},
    {2, 2},
    {2, 2},
    {2, 2},
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
                tensors_x[i][j][k] = NULL;
                tensors_y[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
            }

            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors_x[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    torch_tensors_y[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    break;
                case FLOAT64:
                    torch_tensors_x[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    torch_tensors_y[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                tensors_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k], (runtime_t) i, (datatype_t) j);
                tensors_y[i][j][k] = torch_to_tensor(torch_tensors_y[i][j][k], (runtime_t) i, (datatype_t) j);
            }
        }
    }
}

void teardown(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                tensor_destroy(tensors_x[i][j][k]);
                tensor_destroy(tensors_y[i][j][k]);
                tensor_destroy(returned_tensors[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
            }
        }
    }
    error_print(error);
    error_destroy(error);
}

void test_binary(runtime_binary_elementwise_type_t runtime_binary_elementwise_type)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;

                switch (runtime_binary_elementwise_type)
                {
                case RUNTIME_ADDITION:
                    expected_tensor = torch::add(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case RUNTIME_SUBTRACTION:
                    expected_tensor = torch::subtract(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case RUNTIME_MULTIPLICATION:
                    expected_tensor = torch::mul(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case RUNTIME_DIVISION:
                    expected_tensor = torch::div(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case RUNTIME_POWER:
                    expected_tensor = torch::pow(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case RUNTIME_COMPARE_EQUAL:
                    expected_tensor = torch::eq(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]).to(torch_tensors_x[i][j][k].dtype());
                    break;
                case RUNTIME_COMPARE_GREATER:
                    expected_tensor = torch::gt(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]).to(torch_tensors_x[i][j][k].dtype());
                    break;
                default:
                    ck_abort_msg("unknown binary type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);
                returned_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (runtime_binary_elementwise_type)
                {
                case RUNTIME_ADDITION:
                    error = runtime_addition(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_SUBTRACTION:
                    error = runtime_subtraction(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_MULTIPLICATION:
                    error = runtime_multiplication(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_DIVISION:
                    error = runtime_division(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_POWER:
                    error = runtime_power(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_COMPARE_EQUAL:
                    error = runtime_compare_equal(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_COMPARE_GREATER:
                    error = runtime_compare_greater(tensors_x[i][j][k]->buffer, tensors_y[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                default:
                    ck_abort_msg("unknown binary type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_tensors[i][j][k]->buffer, expected_tensors[i][j][k]->buffer);
            }
        }
    }
}

START_TEST(test_addition)
{
    test_binary(RUNTIME_ADDITION);
}
END_TEST

START_TEST(test_subtraction)
{
    test_binary(RUNTIME_SUBTRACTION);
}
END_TEST

START_TEST(test_multiplication)
{
    test_binary(RUNTIME_MULTIPLICATION);
}
END_TEST

START_TEST(test_division)
{
    test_binary(RUNTIME_DIVISION);
}
END_TEST

START_TEST(test_power)
{
    test_binary(RUNTIME_POWER);
}
END_TEST

START_TEST(test_compare_equal)
{
    test_binary(RUNTIME_COMPARE_EQUAL);
}
END_TEST

START_TEST(test_compare_greater)
{
    test_binary(RUNTIME_ADDITION);
}
END_TEST

Suite *make_buffer_binary_suite(void)
{
    Suite *s;
    TCase *tc_binary;

    s = suite_create("Test Buffer Binary Suite");

    tc_binary = tcase_create("Buffer Binary Case");
    tcase_add_checked_fixture(tc_binary, setup, teardown);
    tcase_add_test(tc_binary, test_addition);
    tcase_add_test(tc_binary, test_subtraction);
    tcase_add_test(tc_binary, test_multiplication);
    tcase_add_test(tc_binary, test_division);
    tcase_add_test(tc_binary, test_power);
    tcase_add_test(tc_binary, test_compare_equal);
    tcase_add_test(tc_binary, test_compare_greater);

    suite_add_tcase(s, tc_binary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_binary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
