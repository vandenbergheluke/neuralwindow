#include <iostream>
#include <tuple>
extern "C"
{
#include <buffer.h>
#include <tensor.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <measure.h>
#include <check.h>
}
#include <test_helper.h>

#define CASES 8

#define MEASUREMENT_ITERS 15

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Device device_cuda(torch::kCUDA);
torch::Device device_cpu(torch::kCPU);

std::vector<int64_t> shapes[CASES] = {
    {2,   2},
    {2,   2},
    {3,   3},
    {3,   3},
    {32,  32},
    {32,  32},
    {128, 128},
    {128, 128},
};

uint64_t axis[CASES] = {
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
};

bool_t keep_dimension[CASES] = {
    true,
    true,
    true,
    true,
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
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    // We're using CPU pytorch because we use an unsupported
                    // version of CUDA... CUDA tests are disabled right now.
                    switch ((datatype_t) j)
                    {
                    case FLOAT32:
                        torch_tensors[i][j][k][z] = torch::randn(shapes[k],
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    case FLOAT64:
                        torch_tensors[i][j][k][z] = torch::randn(shapes[k],
                                torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }

                    tensors[i][j][k][z] = torch_to_tensor(torch_tensors[i][j][k][z], (runtime_t) i, (datatype_t) j);
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
                    tensor_destroy(tensors[i][j][k][z]);
                    tensor_destroy(returned_tensors[i][j][k][z]);
                }
            }
        }
    }
    error_print(error);
    error_destroy(error);
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


START_TEST(test_summation_computational_performance)
{
    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;

    printf("---------------------   Summation   ----------------------\n");

    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t torch_flops_mkl = 0, torch_flops_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
        float64_t nw_flops_mkl = 0, nw_flops_openblas = 0, nw_flops_cuda = 0;
        uint64_t n = shapes[k][0];
        uint64_t num_flop = pow(n, 2);

        printf("Dimensions (%lu, %lu), Axis %lu:\n", n, n, axis[k]);

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
                    torch::Tensor expected_tensor = torch::sum(torch_tensors[i][j][k][z], std::vector<int64_t>({(int64_t) axis[k]}), keep_dimension[k]);
                    torch_end = get_time_nanoseconds();

                    returned_tensors[i][j][k][z] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                    nw_start = get_time_nanoseconds();
                    error = runtime_summation(tensors[i][j][k][z]->buffer, returned_tensors[i][j][k][z]->buffer, axis[k]);
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
END_TEST

START_TEST(test_maximum_computational_performance)
{
    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;

    printf("----------------------   Maximum   -----------------------\n");

    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t torch_flops_mkl = 0, torch_flops_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
        float64_t nw_flops_mkl = 0, nw_flops_openblas = 0, nw_flops_cuda = 0;
        uint64_t n = shapes[k][0];
        uint64_t num_flop = pow(n, 2);

        printf("Dimensions (%lu, %lu), Axis %lu:\n", n, n, axis[k]);

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
                    torch::Tensor expected_tensor = std::get<0>(torch::max(torch_tensors[i][j][k][z], 
                                                  {(int64_t) axis[k]}, keep_dimension[k]));
                    torch_end = get_time_nanoseconds();

                    returned_tensors[i][j][k][z] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                    nw_start = get_time_nanoseconds();
                    error = runtime_maximum(tensors[i][j][k][z]->buffer, returned_tensors[i][j][k][z]->buffer, axis[k]);
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
END_TEST

Suite *make_buffer_reduction_perf_suite(void)
{
    Suite *s;
    TCase *tc_reduction;

    s = suite_create("Test Buffer Reduction Performance Suite");

    tc_reduction = tcase_create("Buffer Reduction Performance Case");
    tcase_add_checked_fixture(tc_reduction, setup, teardown);
    tcase_add_test(tc_reduction, test_summation_computational_performance);
    tcase_add_test(tc_reduction, test_maximum_computational_performance);

    suite_add_tcase(s, tc_reduction);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_reduction_perf_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}