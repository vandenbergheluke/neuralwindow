#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <measure.h>
}
#include <torch/torch.h>

#define CASES 4

#define MEASUREMENT_ITERS 15

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

view_t *views[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
view_t *returned_views[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Tensor tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

std::vector<int64_t> shapes[CASES] = {
    {1, 1},
    {2, 2},
    {3, 3},
    {4, 4},
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
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    buffers[i][j][k][z] = NULL;
                    returned_buffers[i][j][k][z] = NULL;

                    views[i][j][k][z] = NULL;
                    returned_views[i][j][k][z] = NULL;
                }
            }

            
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    switch ((datatype_t) j)
                    {
                    case FLOAT32:
                        tensors[i][j][k][z] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                        break;
                    case FLOAT64:
                        tensors[i][j][k][z] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }

                    error = view_create(&views[i][j][k][z], 
                                        (uint64_t) tensors[i][j][k][z].storage_offset(),
                                        (uint64_t) tensors[i][j][k][z].ndimension(),
                                        (uint64_t *) tensors[i][j][k][z].sizes().data(),
                                        NULL);
                    ck_assert_ptr_null(error);
                    error = buffer_create(&buffers[i][j][k][z],
                                          (runtime_t) i,
                                          (datatype_t) j,
                                          views[i][j][k][z],
                                          (void *) tensors[i][j][k][z].data_ptr(),
                                          (uint64_t) tensors[i][j][k][z].numel(),
                                          true);
                    ck_assert_ptr_null(error);

                    error = view_create(&returned_views[i][j][k][z],
                                        (uint64_t) tensors[i][j][k][z].storage_offset(),
                                        (uint64_t) tensors[i][j][k][z].ndimension(),
                                        (uint64_t *) tensors[i][j][k][z].sizes().data(),
                                        NULL);
                    ck_assert_ptr_null(error);
                    error = buffer_create(&returned_buffers[i][j][k][z],
                                          (runtime_t) i,
                                          (datatype_t) j,
                                          returned_views[i][j][k][z],
                                          NULL,
                                          (uint64_t) tensors[i][j][k][z].numel(),
                                          true);
                    ck_assert_ptr_null(error);
                }
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
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    buffer_destroy(buffers[i][j][k][z]);
                    buffer_destroy(returned_buffers[i][j][k][z]);
                }
            }
        }
    }
    error_print(error);
    error_destroy(error);
}

START_TEST(test_exponential_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::exp(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_exponential(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch exponential performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW exponential performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_logarithm_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::log(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_logarithm(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch logarithm performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW logarithm performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_sine_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::sin(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_sine(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch sine performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW sine performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_cosine_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::cos(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_cosine(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch cosine performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW cosine performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_square_root_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::sqrt(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_square_root(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch square root performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW square root performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_reciprocal_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t torch_avg_flops = 0;
    float64_t nw_avg_perf = 0;
    float64_t nw_avg_flops = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t n = ((uint64_t *) tensors[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = n;
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = UNARY_PERF_CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::reciprocal(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_reciprocal(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    torch_avg_flops += ((float64_t) num_flop / (float64_t) torch_completion_time) / total_runs;
                    nw_avg_perf += (float64_t) nw_completion_time / total_runs;
                    nw_avg_flops += ((float64_t) num_flop / (float64_t) nw_completion_time) / total_runs;
                }
            }
        }
    }

    printf("PyTorch reciprocal performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW reciprocal performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n", nw_avg_flops);
}
END_TEST

START_TEST(test_copy_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::clone(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_copy(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch copy performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW copy performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_contiguous_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = tensors[i][j][k][z].contiguous();
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_contiguous(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch contiguous performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW contiguous performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_negation_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::neg(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_negation(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = torch_end - torch_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_avg_perf / total_runs;
                }
            }
        }
    }

    printf("PyTorch negation performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW negation performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_rectified_linear_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t torch_avg_flops = 0;
    float64_t nw_avg_perf = 0;
    float64_t nw_avg_flops = 0;
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t n = ((uint64_t *) tensors[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = n;
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = UNARY_PERF_CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::relu(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_rectified_linear(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    torch_avg_flops += ((float64_t) num_flop / (float64_t) torch_completion_time) / total_runs;
                    nw_avg_perf += (float64_t) nw_completion_time / total_runs;
                    nw_avg_flops += ((float64_t) num_flop / (float64_t) nw_completion_time) / total_runs;
                }
            }
        }
    }

    printf("PyTorch rectified linear performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW rectified linear performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n", nw_avg_flops);
}
END_TEST

Suite *make_buffer_unary_perf_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Buffer Unary Performance Suite");

    // Unary Performance Operations
    tc_unary = tcase_create("Buffer Unary Case");
    tcase_add_checked_fixture(tc_unary, setup, teardown);
    tcase_add_test(tc_unary, test_exponential_computational_performance);
    tcase_add_test(tc_unary, test_logarithm_computational_performance);
    tcase_add_test(tc_unary, test_sine_computational_performance);
    tcase_add_test(tc_unary, test_cosine_computational_performance);
    tcase_add_test(tc_unary, test_square_root_computational_performance);
    tcase_add_test(tc_unary, test_reciprocal_computational_performance);
    tcase_add_test(tc_unary, test_copy_computational_performance);
    tcase_add_test(tc_unary, test_contiguous_computational_performance);
    tcase_add_test(tc_unary, test_negation_computational_performance);
    tcase_add_test(tc_unary, test_rectified_linear_computational_performance);

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_unary_perf_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
