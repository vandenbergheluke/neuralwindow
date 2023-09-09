#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <test_helper.h>
}
#include <torch/torch.h>

#define CASES 4

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

torch::Tensor tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    {2, 2},
    {2, 2},
    {2, 2},
    {2, 2},
};

uint64_t axis[CASES] = {
    0,
    0,
    1,
    1,
};

bool_t keep_dimension[CASES] = {
    true,
    true,
    true,
    true,
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
                buffers[i][j][k] = NULL;
                returned_buffers[i][j][k] = NULL;
                expected_buffers[i][j][k] = NULL;
            }
            
            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    tensors[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    break;
                case FLOAT64:
                    tensors[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                view_t *view;
                storage_t *storage;

                error = view_create(&view, 
                                    (uint64_t) tensors[i][j][k].storage_offset(),
                                    (uint64_t) tensors[i][j][k].ndimension(),
                                    (uint64_t *) tensors[i][j][k].sizes().data(),
                                    (uint64_t *) tensors[i][j][k].strides().data());
                ck_assert_ptr_null(error);
                error = storage_create(&storage,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      tensors[i][j][k].nbytes() / 
                                      datatype_size((datatype_t) j),
                                      (void *) tensors[i][j][k].data_ptr());
                error = buffer_create(&buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);
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
                buffer_destroy(buffers[i][j][k]);
                buffer_destroy(returned_buffers[i][j][k]);
                buffer_destroy(expected_buffers[i][j][k]);
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
                    expected_tensor = torch::sum(tensors[i][j][k], 
                                                 std::vector<int64_t>({(int64_t) axis[k]}),
                                                 keep_dimension[k]);
                    break;
                case RUNTIME_MAXIMUM:
                    expected_tensor = std::get<0>(torch::max(tensors[i][j][k], 
                                                  {(int64_t) axis[k]}, keep_dimension[k]));
                    break;
                default:
                    ck_abort_msg("unknown reduction type.");
                }

                view_t *view;
                storage_t *storage;

                error = view_create(&view,
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    (uint64_t *) expected_tensor.strides().data());
                ck_assert_ptr_null(error);
                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       expected_tensor.nbytes() / 
                                       datatype_size((datatype_t) j),
                                       NULL);
                error = buffer_create(&returned_buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
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
                                      expected_tensor.nbytes() / 
                                      datatype_size((datatype_t) j),
                                      (void *) expected_tensor.data_ptr());
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                switch (runtime_reduction_type)
                {
                case RUNTIME_SUMMATION:
                    error = runtime_summation(buffers[i][j][k], returned_buffers[i][j][k], axis[k]);
                    break;
                case RUNTIME_MAXIMUM:
                    error = runtime_maximum(buffers[i][j][k], returned_buffers[i][j][k], axis[k]);
                    break;
                default:
                    ck_abort_msg("unknown reduction type.");
                }
                ck_assert_ptr_null(error);
                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
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
