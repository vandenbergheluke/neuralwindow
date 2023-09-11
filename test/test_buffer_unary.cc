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

#define CASES 6

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    {1, 2},
    {1, 2},
    {1, 2},
    {1, 2},
    {3, 4, 5},
    {3, 4, 5},
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
                tensors[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
            }

            
            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    break;
                case FLOAT64:
                    torch_tensors[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                tensors[i][j][k] = torch_to_tensor(torch_tensors[i][j][k], (runtime_t) i, (datatype_t) j);
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
                tensor_destroy(tensors[i][j][k]);
                tensor_destroy(returned_tensors[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
            }
        }
    }
    error_print(error);
    error_destroy(error);
}

void test_unary(runtime_unary_type_t runtime_unary_type)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;

                switch (runtime_unary_type)
                {
                case RUNTIME_EXPONENTIAL:
                    expected_tensor = torch::exp(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_LOGARITHM:
                    expected_tensor = torch::log(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_SINE:
                    expected_tensor = torch::sin(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_COSINE:
                    expected_tensor = torch::cos(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_SQUARE_ROOT:
                    expected_tensor = torch::sqrt(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_RECIPROCAL:
                    expected_tensor = torch::reciprocal(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_NEGATION:
                    expected_tensor = torch::neg(torch_tensors[i][j][k]);
                    break;
                case RUNTIME_CONTIGUOUS:
                    expected_tensor = torch_tensors[i][j][k].contiguous();
                    break;
                case RUNTIME_RECTIFIED_LINEAR:
                    expected_tensor = torch::relu(torch_tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);
                returned_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (runtime_unary_type)
                {
                case RUNTIME_EXPONENTIAL:
                    error = runtime_exponential(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_LOGARITHM:
                    error = runtime_logarithm(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_SINE:
                    error = runtime_sine(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_COSINE:
                    error = runtime_cosine(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_SQUARE_ROOT:
                    error = runtime_square_root(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_RECIPROCAL:
                    error = runtime_reciprocal(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_NEGATION:
                    error = runtime_negation(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_CONTIGUOUS:
                    error = runtime_contiguous(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                case RUNTIME_RECTIFIED_LINEAR:
                    error = runtime_rectified_linear(tensors[i][j][k]->buffer, returned_tensors[i][j][k]->buffer);
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_tensors[i][j][k]->buffer, expected_tensors[i][j][k]->buffer);
            }
        }
    }
}

START_TEST(test_exponential)
{
    test_unary(RUNTIME_EXPONENTIAL);
}
END_TEST

START_TEST(test_logarithm)
{
    test_unary(RUNTIME_LOGARITHM);
}
END_TEST

START_TEST(test_sine)
{
    test_unary(RUNTIME_SINE);
}
END_TEST

START_TEST(test_cosine)
{
    test_unary(RUNTIME_COSINE);
}
END_TEST

START_TEST(test_square_root)
{
    test_unary(RUNTIME_SQUARE_ROOT);
}
END_TEST

START_TEST(test_reciprocal)
{
    test_unary(RUNTIME_RECIPROCAL);
}
END_TEST


START_TEST(test_contiguous)
{
    test_unary(RUNTIME_CONTIGUOUS);
}
END_TEST

START_TEST(test_negation)
{
    test_unary(RUNTIME_NEGATION);
}
END_TEST

START_TEST(test_rectified_linear)
{
    test_unary(RUNTIME_RECTIFIED_LINEAR);
}
END_TEST

Suite *make_buffer_unary_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Buffer Unary Suite");

    // Unary Operations
    tc_unary = tcase_create("Buffer Unary Case");
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

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_unary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
