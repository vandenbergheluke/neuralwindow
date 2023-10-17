#include <iostream>
extern "C"
{
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <datatype.h>
#include <errors.h>
#include <measure.h>
#include <check.h>
#include <test_helper.h>
}
#include <test_helper_torch.h>

#define CASES 5

#define MEASUREMENT_ITERS 15

nw_error_t *error;

tensor_t *tensors_x[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS] = { NULL };
tensor_t *tensors_y[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS] = { NULL };
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS] = { NULL };

torch::Tensor torch_tensors_x[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
torch::Tensor torch_tensors_y[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Device device_cuda(torch::kCUDA);
torch::Device device_cpu(torch::kCPU);

std::vector<int64_t> shapes[CASES] = {
    {1,   1},
    {2,   2},
    {3,   3},
    {32,  32},
    {128, 128},
};

std::vector<int64_t> expanded_shapes[CASES] = {
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
                    returned_tensors[i][j][k][z] = NULL;

                    switch ((datatype_t) j)
                    {
                    case FLOAT32:
                        torch_tensors_x[i][j][k][z] = torch::randn(shapes[k], 
                                                              torch::TensorOptions()
                                                              .dtype(torch::kFloat32)
                                                              // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                                              ).expand(expanded_shapes[k]);
                        torch_tensors_y[i][j][k][z] = torch::randn(shapes[k], 
                                                              torch::TensorOptions()
                                                              .dtype(torch::kFloat32)
                                                              // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                                              ).expand(expanded_shapes[k]);
                        break;
                    case FLOAT64:
                        torch_tensors_x[i][j][k][z] = torch::randn(shapes[k],
                                                              torch::TensorOptions()
                                                              .dtype(torch::kFloat64)
                                                              // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                                              ).expand(expanded_shapes[k]);
                        torch_tensors_y[i][j][k][z] = torch::randn(shapes[k],
                                                              torch::TensorOptions()
                                                              .dtype(torch::kFloat64)
                                                              // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                                              ).expand(expanded_shapes[k]);
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }

                    tensors_x[i][j][k][z] = torch_to_tensor(torch_tensors_x[i][j][k][z], (runtime_t) i , (datatype_t) j);
                    tensors_y[i][j][k][z] = torch_to_tensor(torch_tensors_y[i][j][k][z], (runtime_t) i , (datatype_t) j);
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
                    tensor_destroy(tensors_x[i][j][k][z]);
                    tensor_destroy(tensors_y[i][j][k][z]);
                    tensor_destroy(returned_tensors[i][j][k][z]);
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

void performance_test(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> torch_op,
        std::function<nw_error_t *(tensor_t *, tensor_t *, tensor_t **)> nw_op)
{
    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;
    
    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
        int64_t n = shapes[k][0];

        printf("Dimensions (%ld, %ld):\n", n, n);

        for (int i = 0; i < RUNTIMES; ++i)
        {
            // Take average time of DATATYPES * MEASUREMENT_ITERS iterations for
            // each runtime.
            for (int j = 0; j < DATATYPES; ++j)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    int64_t torch_start, torch_end;
                    int64_t torch_completion_time;
                    int64_t nw_start, nw_end;
                    int64_t nw_completion_time;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch_op(torch_tensors_x[i][j][k][z], torch_tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = nw_op(tensors_x[i][j][k][z], tensors_y[i][j][k][z], &returned_tensors[i][j][k][z]);
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

void performance_test(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> torch_op,
        std::function<nw_error_t *(tensor_t *, tensor_t *, tensor_t **)> nw_op,
        std::function<int64_t(int64_t)> flop_calc)
{
    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;
    
    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t torch_flops_mkl = 0, torch_flops_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
        float64_t nw_flops_mkl = 0, nw_flops_openblas = 0, nw_flops_cuda = 0;
        int64_t n = shapes[k][0];
        int64_t num_flop = flop_calc(n);

        printf("Dimensions (%ld, %ld):\n", n, n);

        for (int i = 0; i < RUNTIMES; ++i)
        {
            // Take average time of DATATYPES * MEASUREMENT_ITERS iterations for
            // each runtime.
            for (int j = 0; j < DATATYPES; ++j)
            {
                for (int z = 0; z < MEASUREMENT_ITERS; ++z)
                {
                    int64_t torch_start, torch_end;
                    int64_t torch_completion_time;
                    int64_t nw_start, nw_end;
                    int64_t nw_completion_time;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch_op(torch_tensors_x[i][j][k][z], torch_tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = nw_op(tensors_x[i][j][k][z], tensors_y[i][j][k][z], &returned_tensors[i][j][k][z]);
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

START_TEST(test_addition_computational_performance)
{
    printf("----------------------   Addition   ----------------------\n");
    performance_test(AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (int64_t n) -> int64_t { return pow(n, 2); });
}
END_TEST

START_TEST(test_subtraction_computational_performance)
{
    printf("--------------------   Subtraction   ---------------------\n");
    performance_test(AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (int64_t n) -> int64_t { return pow(n, 2); });
}
END_TEST

START_TEST(test_multiplication_computational_performance)
{
    printf("-------------------   Multiplication   -------------------\n");
    performance_test(AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (int64_t n) -> int64_t { return pow(n, 2); });
}
END_TEST

START_TEST(test_division_computational_performance)
{
    printf("----------------------   Division   ----------------------\n");
    performance_test(AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (int64_t n) -> int64_t { return pow(n, 2); });
}
END_TEST

START_TEST(test_power_computational_performance)
{
    printf("-----------------------   Power   ------------------------\n");
    performance_test(AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
}
END_TEST

START_TEST(test_matrix_multiplication_computational_performance)
{
    printf("---------------   Matrix Multiplication   ----------------\n");
    performance_test(AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (int64_t n) -> int64_t {
            return ((2.0 / 3.0) * (float64_t) pow(n, 3))
            + (2.0 * (float64_t) pow(n, 2));
            });
}
END_TEST

Suite *make_buffer_unary_perf_suite(void)
{
    Suite *s;
    TCase *tc_binary;

    s = suite_create("Test Tensor Binary Performance Suite");

    // Binary Performance Operations
    tc_binary = tcase_create("Tensor Binary Case");
    tcase_add_checked_fixture(tc_binary, setup, teardown);
    tcase_add_test(tc_binary, test_addition_computational_performance);
    tcase_add_test(tc_binary, test_subtraction_computational_performance);
    tcase_add_test(tc_binary, test_multiplication_computational_performance);
    tcase_add_test(tc_binary, test_division_computational_performance);
    tcase_add_test(tc_binary, test_power_computational_performance);
    tcase_add_test(tc_binary, test_matrix_multiplication_computational_performance);

    suite_add_tcase(s, tc_binary);

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
