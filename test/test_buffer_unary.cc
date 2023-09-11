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

#define CASES 6

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

torch::Tensor tensors[RUNTIMES][DATATYPES][CASES];

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
                                      tensors[i][j][k].storage().nbytes() / 
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

void test_unary(runtime_unary_type_t runtime_unary_type)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;
                view_t *view;
                storage_t *storage;

                switch (runtime_unary_type)
                {
                case RUNTIME_EXPONENTIAL:
                    expected_tensor = torch::exp(tensors[i][j][k]);
                    break;
                case RUNTIME_LOGARITHM:
                    expected_tensor = torch::log(tensors[i][j][k]);
                    break;
                case RUNTIME_SINE:
                    expected_tensor = torch::sin(tensors[i][j][k]);
                    break;
                case RUNTIME_COSINE:
                    expected_tensor = torch::cos(tensors[i][j][k]);
                    break;
                case RUNTIME_SQUARE_ROOT:
                    expected_tensor = torch::sqrt(tensors[i][j][k]);
                    break;
                case RUNTIME_RECIPROCAL:
                    expected_tensor = torch::reciprocal(tensors[i][j][k]);
                    break;
                case RUNTIME_NEGATION:
                    expected_tensor = torch::neg(tensors[i][j][k]);
                    break;
                case RUNTIME_CONTIGUOUS:
                    expected_tensor = tensors[i][j][k].contiguous();
                    break;
                case RUNTIME_RECTIFIED_LINEAR:
                    expected_tensor = torch::relu(tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }

                error = view_create(&view,
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    (uint64_t *) expected_tensor.strides().data());
                ck_assert_ptr_null(error);
                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       expected_tensor.storage().nbytes() / 
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
                                      expected_tensor.storage().nbytes() / 
                                      datatype_size((datatype_t) j),
                                      (void *) expected_tensor.data_ptr());
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                switch (runtime_unary_type)
                {
                case RUNTIME_EXPONENTIAL:
                    error = runtime_exponential(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_LOGARITHM:
                    error = runtime_logarithm(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_SINE:
                    error = runtime_sine(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_COSINE:
                    error = runtime_cosine(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_SQUARE_ROOT:
                    error = runtime_square_root(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_RECIPROCAL:
                    error = runtime_reciprocal(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_NEGATION:
                    error = runtime_negation(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_CONTIGUOUS:
                    error = runtime_contiguous(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RUNTIME_RECTIFIED_LINEAR:
                    error = runtime_rectified_linear(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
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
