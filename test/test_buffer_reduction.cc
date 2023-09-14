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

#define CASES 8

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> axis[CASES] = {
    {0},
    {0},
    {1},
    {1},
    {0, 2},
    {1},
    {0, 1, 2, 3},
    {0, 2, 3},
};

bool_t keep_dimension[CASES] = {
    false,
    true,
    false,
    true,
    true,
    false,
    false,
    true,
};

std::vector<int64_t> shapes[CASES] = {
    {1},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
    {2, 3, 4, 5},
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

void test_reduction(runtime_reduction_type_t runtime_reduction_type)
{  
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;

                switch (runtime_reduction_type)
                {
                case RUNTIME_SUMMATION:
                    expected_tensor = torch::sum(torch_tensors[i][j][k], axis[k], keep_dimension[k]);
                    break;
                case RUNTIME_MAXIMUM:
                    expected_tensor = torch::amax(torch_tensors[i][j][k], axis[k], keep_dimension[k]);
                    break;
                default:
                    ck_abort_msg("unknown reduction type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);
                returned_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (runtime_reduction_type)
                {
                case RUNTIME_SUMMATION:
                    error = runtime_summation(tensors[i][j][k]->buffer,
                                              (uint64_t *) axis[k].data(),
                                              (uint64_t) axis[k].size(),
                                              returned_tensors[i][j][k]->buffer,
                                              keep_dimension[k]);
                    break;
                case RUNTIME_MAXIMUM:
                    error = runtime_maximum(tensors[i][j][k]->buffer,
                                            (uint64_t *) axis[k].data(),
                                            (uint64_t) axis[k].size(),
                                            returned_tensors[i][j][k]->buffer,
                                            keep_dimension[k]);
                    break;
                default:
                    ck_abort_msg("unknown reduction type.");
                }

                ck_assert_ptr_null(error);
                ck_assert_buffer_eq(returned_tensors[i][j][k]->buffer, expected_tensors[i][j][k]->buffer);
            }
        }
    }
}

START_TEST(test_summation)
{
    test_reduction(RUNTIME_SUMMATION);
}
END_TEST

START_TEST(test_maximum)
{
    test_reduction(RUNTIME_MAXIMUM);
}
END_TEST

Suite *make_buffer_reduction_suite(void)
{
    Suite *s;
    TCase *tc_reduction;

    s = suite_create("Test Buffer Reduction Suite");

    tc_reduction = tcase_create("Buffer Reduction Case");
    tcase_add_checked_fixture(tc_reduction, setup, teardown);
    tcase_add_test(tc_reduction, test_summation);
    tcase_add_test(tc_reduction, test_maximum);

    suite_add_tcase(s, tc_reduction);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_reduction_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
