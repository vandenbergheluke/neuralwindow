// Unary operation performance tests. We're losing some accuracy due to lambdas
// adding time to pytorch operations but I gather we're more concerned with
// being multiplicatively faster/slower than we are with counting individual
// clock cycles.
//
// TODO: Find a generic solution that would not result in issues due to
// overloading if pytorch changed its headers!!!

#include <cstdio>
#include <cmath>
#include <iostream>
#include <tuple>
extern "C"
{
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <measure.h>

#include <test_helper.h>

#include <check.h>
}
#include <torch/torch.h>

#define CASES 5

#define MEASUREMENT_ITERS 15

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

view_t *views[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
view_t *returned_views[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

storage_t *storages[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
storage_t *returned_storages[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Tensor tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Device device_cuda(torch::kCUDA);
torch::Device device_cpu(torch::kCPU);

std::vector<int64_t> shapes[CASES] = {
    {1,   1},
    {2,   2},
    {3,   3},
    {32,  32},
    {128, 128},
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
                    // We're using CPU pytorch because we use an unsupported
                    // version of CUDA... CUDA tests are disabled right now.
                    switch ((datatype_t) j)
                    {
                    case FLOAT32:
                        tensors[i][j][k][z] = torch::randn(shapes[k],
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    case FLOAT64:
                        tensors[i][j][k][z] = torch::randn(shapes[k],
                                torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
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
                    error = storage_create(&storages[i][j][k][z],
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) tensors[i][j][k][z].numel(),
                                           (void *) tensors[i][j][k][z].data_ptr());
                    ck_assert_ptr_null(error);
                    error = buffer_create(&buffers[i][j][k][z],
                                          views[i][j][k][z],
                                          storages[i][j][k][z],
                                          false);
                    ck_assert_ptr_null(error);

                    error = view_create(&returned_views[i][j][k][z],
                                        (uint64_t) tensors[i][j][k][z].storage_offset(),
                                        (uint64_t) tensors[i][j][k][z].ndimension(),
                                        (uint64_t *) tensors[i][j][k][z].sizes().data(),
                                        NULL);
                    ck_assert_ptr_null(error);
                    error = storage_create(&returned_storages[i][j][k][z],
                                           (runtime_t) i,
                                           (datatype_t) j,
                                           (uint64_t) tensors[i][j][k][z].numel(),
                                           (void *) tensors[i][j][k][z].data_ptr());
                    ck_assert_ptr_null(error);
                    error = buffer_create(&returned_buffers[i][j][k][z],
                                          returned_views[i][j][k][z],
                                          returned_storages[i][j][k][z],
                                          false);
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

void print_heuristics(float64_t torch_time_mkl, float64_t torch_time_cuda,
        float64_t nw_time_mkl, float64_t nw_time_openblas,
        float64_t nw_time_cuda)
{
    printf("MKL:\n");
    printf("PyTorch performance (nsec): %0.2lf\n", torch_time_mkl);
    printf("NW performance (nsec): %0.2lf\n", nw_time_mkl);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_time_mkl / torch_time_mkl);
    printf("OpenBLAS:\n");
    printf("NW performance (nsec): %0.2lf\n", nw_time_openblas);
    printf("CUDA:\n");
    // printf("PyTorch performance (nsec): %0.2lf\n", torch_time_cuda);
    printf("NW performance (nsec): %0.2lf\n\n", nw_time_cuda);
    // printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n\n", nw_time_cuda / torch_time_cuda);
}

void print_heuristics(float64_t torch_time_mkl, float64_t torch_flops_mkl,
        float64_t torch_time_cuda, float64_t torch_flops_cuda,
        float64_t nw_time_mkl, float64_t nw_flops_mkl,
        float64_t nw_time_openblas, float64_t nw_flops_openblas,
        float64_t nw_time_cuda, float64_t nw_flops_cuda)
{
    printf("MKL:\n");
    printf("PyTorch performance: %0.2lf nsec, %0.2lf FLOPS\n", torch_time_mkl, torch_flops_mkl);
    printf("NW exponential performance: %0.2lf nsec, %0.2lf FLOPS\n", nw_time_mkl, nw_flops_mkl);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n\n", nw_time_mkl / torch_time_mkl);
    printf("OpenBLAS:\n");
    printf("NW exponential performance: %0.2lf nsec, %0.2lf FLOPS\n", nw_time_openblas, nw_flops_openblas);
    printf("CUDA:\n");
    // printf("PyTorch performance: %0.2lf nsec, %0.2lf FLOPS\n", torch_time_cuda, torch_flops_cuda);
    printf("NW exponential performance: %0.2lf nsec, %0.2lf FLOPS\n\n", nw_time_cuda, nw_flops_cuda);
    // printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n\n", nw_time_cuda / torch_time_cuda);
}

void performance_test(std::function<torch::Tensor(torch::Tensor)> torch_op,
        std::function<nw_error_t *(buffer_t *, buffer_t *)> nw_op)
{
    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;
    
    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
        uint64_t n = shapes[k][0];

        printf("Dimensions (%lu, %lu):\n", n, n);

        for (int i = 0; i < RUNTIMES; ++i)
        {
            // Take average time of DATATYPES * MEASUREMENT_ITERS iterations for
            // each runtime.
            for (int j = 0; j < DATATYPES; ++j)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch_op(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = nw_op(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    switch ((runtime_t) i)
                    {
                        case OPENBLAS_RUNTIME:
                            // Pytorch uses MKL on CPU

                            nw_time_openblas += (float64_t) nw_completion_time / total_runs;
                            break;
                        case MKL_RUNTIME:
                            // Torch MKL gets double the runs as a biproduct of
                            // how the tests are setup.

                            torch_time_mkl += (float64_t) torch_completion_time / (2 * total_runs);
                            nw_time_mkl += (float64_t) nw_completion_time / total_runs;
                            break;
                        case CU_RUNTIME:
                            torch_time_cuda += (float64_t) torch_completion_time / total_runs;
                            nw_time_cuda += (float64_t) nw_completion_time / total_runs;
                            break;
                        default:
                        ck_abort_msg("unknown runtime.");
                    }
                }

            }
        }

        print_heuristics(torch_time_mkl, torch_time_cuda, nw_time_mkl,
                nw_time_openblas, nw_time_cuda);
    }
}

void performance_test(std::function<torch::Tensor(torch::Tensor)> torch_op,
        std::function<nw_error_t *(buffer_t *, buffer_t *)> nw_op,
        std::function<uint64_t(uint64_t)> flop_calc)
{
    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;
    
    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t torch_flops_mkl = 0, torch_flops_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
        float64_t nw_flops_mkl = 0, nw_flops_openblas = 0, nw_flops_cuda = 0;
        uint64_t n = shapes[k][0];
        uint64_t num_flop = flop_calc(n);

        printf("Dimensions (%lu, %lu):\n", n, n);

        for (int i = 0; i < RUNTIMES; ++i)
        {
            // Take average time of DATATYPES * MEASUREMENT_ITERS iterations for
            // each runtime.
            for (int j = 0; j < DATATYPES; ++j)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch_op(tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = nw_op(buffers[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    switch ((runtime_t) i)
                    {
                        case OPENBLAS_RUNTIME:
                            // Pytorch uses MKL on CPU

                            nw_time_openblas += (float64_t) nw_completion_time / total_runs;
                            nw_flops_openblas += ((float64_t) num_flop * 1000000000) / ((float64_t) nw_completion_time * total_runs);
                            break;
                        case MKL_RUNTIME:
                            // Torch MKL gets double the runs as a biproduct of
                            // how the tests are setup.

                            torch_time_mkl += (float64_t) torch_completion_time / (2 * total_runs);
                            torch_flops_mkl += ((float64_t) num_flop * 1000000000) / ((float64_t) torch_completion_time * 2 * total_runs);
                            nw_time_mkl += (float64_t) nw_completion_time / total_runs;
                            nw_flops_mkl += ((float64_t) num_flop * 1000000000) / ((float64_t) nw_completion_time * total_runs);
                            break;
                        case CU_RUNTIME:
                            torch_time_cuda += (float64_t) torch_completion_time / total_runs;
                            torch_flops_cuda += ((float64_t) num_flop * 1000000000) / ((float64_t) torch_completion_time * total_runs);
                            nw_time_cuda += (float64_t) nw_completion_time / total_runs;
                            nw_flops_cuda += ((float64_t) num_flop * 1000000000) / ((float64_t) nw_completion_time * total_runs);
                            break;
                        default:
                        ck_abort_msg("unknown runtime.");
                    }
                }

            }
        }

        print_heuristics(torch_time_mkl, torch_flops_mkl, torch_time_cuda,
                torch_flops_cuda, nw_time_mkl, nw_flops_mkl, nw_time_openblas,
                nw_flops_openblas, nw_time_cuda, nw_flops_cuda);
    }
}

START_TEST(test_exponential_computational_performance)
{
    printf("--------------------   Exponential   ---------------------\n");
    performance_test(&torch::exp, &runtime_exponential);
}
END_TEST

START_TEST(test_logarithm_computational_performance)
{
    printf("---------------------   Logarithm   ----------------------\n");
    performance_test(&torch::log, &runtime_logarithm);
}
END_TEST

START_TEST(test_sine_computational_performance)
{
    printf("------------------------   Sine   ------------------------\n");
    performance_test(&torch::sin, &runtime_sine);
}
END_TEST

START_TEST(test_cosine_computational_performance)
{
    printf("-----------------------   Cosine   -----------------------\n");
    performance_test(&torch::cos, &runtime_cosine);
}
END_TEST

START_TEST(test_square_root_computational_performance)
{
    printf("--------------------   Square Root   ---------------------\n");
    performance_test(&torch::sqrt, &runtime_square_root);
}
END_TEST

START_TEST(test_reciprocal_computational_performance)
{
    printf("---------------------   Reciprocal   ---------------------\n");
    performance_test(&torch::reciprocal, &runtime_reciprocal,
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
}
END_TEST

START_TEST(test_copy_computational_performance)
{
    printf("------------------------   Copy   ------------------------\n");
    performance_test([] (torch::Tensor t) -> torch::Tensor {
            return torch::clone(t);
            },
            &runtime_copy);
}
END_TEST

START_TEST(test_contiguous_computational_performance)
{
    printf("---------------------   Contiguous   ---------------------\n");
    performance_test([] (torch::Tensor t) -> torch::Tensor {
            return t.contiguous();
            },
            &runtime_contiguous);
}
END_TEST

START_TEST(test_negation_computational_performance)
{
    printf("----------------------   Negation   ----------------------\n");
    performance_test(&torch::neg, &runtime_negation);
}
END_TEST

START_TEST(test_rectified_linear_computational_performance)
{
    printf("------------------   Rectified Linear   ------------------\n");
    performance_test(&torch::relu, &runtime_rectified_linear,
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
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
