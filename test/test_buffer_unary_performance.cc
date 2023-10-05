#include <iostream>
#include <mgl2/base_cf.h>
#include <mgl2/canvas_cf.h>
#include <vector>
#include <tuple>
#include <string>

#include <mgl2/mgl_cf.h>
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

#define CASES 7

#define MEASUREMENT_ITERS 30

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Device device_cuda(torch::kCUDA);
torch::Device device_cpu(torch::kCPU);

// Min at 0, max at -1 because tests are written lazily for tracking x range.
std::vector<int64_t> shapes[CASES] = {
    {1,   1},
    {2,   2},
    {4,   4},
    {8,   8},
    {32,  32},
    {64,  64},
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
                    returned_tensors[i][j][k][z] = NULL;
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

void plot_heuristics(std::string t, std::string save_path, float64_t* x,
        int x_n, float64_t* y, int y_n)
{
    HMGL graph = mgl_create_graph(800,400);

    HMDT x_mgl = mgl_create_data();
    HMDT y_mgl = mgl_create_data();

    mgl_data_set_double(x_mgl, x, x_n, 1, 1);
    mgl_data_set_double(y_mgl, y, y_n, 1, 1);

    mgl_fill_background(graph, 1, 1, 1, 1);

    //mgl_subplot(graph, 3, 3, 4, "");
    mgl_inplot(graph, 0, 1, 0, 1);
    mgl_title(graph, t.c_str(), "", 5);
    mgl_set_range_dat(graph, 'x', x_mgl, 0);
    mgl_set_range_dat(graph, 'y', y_mgl, 0);
    mgl_axis(graph, "xy", "", "");
    mgl_axis_grid(graph, "xy", "|h", "");
    // TODO: NOT WORKING
    mgl_label(graph, 'x', "Square Matrix Width", 0, "");
    mgl_label(graph, 'y', "Time (nsec)", 0, "");
    mgl_box(graph);
    mgl_plot_xy(graph, x_mgl, y_mgl, "e#.", "");

    mgl_write_png(graph, save_path.c_str(), "w");

    mgl_delete_graph(graph);
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

void performance_test(std::string op_name,
        std::function<torch::Tensor(torch::Tensor)> torch_op,
        std::function<nw_error_t *(buffer_t *, buffer_t *)> nw_op)
{
    float64_t torch_time_arr_mkl[CASES];
    float64_t torch_time_arr_cuda[CASES];

    float64_t nw_time_arr_mkl[CASES];
    float64_t nw_time_arr_openblas[CASES];
    float64_t nw_time_arr_cuda[CASES];

    // Minimum time (for range calculations)
    float64_t torch_time_min_mkl = DBL_MAX;
    float64_t torch_time_min_cuda = DBL_MAX;

    float64_t nw_time_min_mkl = DBL_MAX;
    float64_t nw_time_min_openblas = DBL_MAX;
    float64_t nw_time_min_cuda = DBL_MAX;

    // Maximum time (for range calculations)
    float64_t torch_time_max_mkl = DBL_MIN;
    float64_t torch_time_max_cuda = DBL_MIN;

    float64_t nw_time_max_mkl = DBL_MIN;
    float64_t nw_time_max_openblas = DBL_MIN;
    float64_t nw_time_max_cuda = DBL_MIN;

    // TODO: Figure out a smart way to manage dims.
    float64_t x[CASES];

    uint32_t total_runs = DATATYPES * MEASUREMENT_ITERS;
    
    for (int k = 0; k < CASES; ++k)
    {
        float64_t torch_time_mkl = 0, torch_time_cuda = 0;
        float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;

        uint64_t n = shapes[k][0];

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
                    torch::Tensor expected_tensor = torch_op(torch_tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    returned_tensors[i][j][k][z] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                    nw_start = get_time_nanoseconds();
                    error = nw_op(tensors[i][j][k][z]->buffer, returned_tensors[i][j][k][z]->buffer);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    switch ((runtime_t) i)
                    {
                        case OPENBLAS_RUNTIME:
                            // Pytorch uses MKL on CPU

                            nw_time_openblas += (float64_t) nw_completion_time / total_runs;

                            if (nw_time_min_openblas > nw_completion_time)
                            {
                                nw_time_min_openblas = nw_completion_time;
                            }

                            if (nw_time_max_openblas < nw_completion_time)
                            {
                                nw_time_max_openblas = nw_completion_time;
                            }
                            break;
                        case MKL_RUNTIME:
                            // Torch MKL gets double the runs as a biproduct of
                            // how the tests are setup.

                            torch_time_mkl += (float64_t) torch_completion_time / (2 * total_runs);
                            nw_time_mkl += (float64_t) nw_completion_time / total_runs;


                            if (torch_time_min_mkl > torch_completion_time)
                            {
                                torch_time_min_mkl = torch_completion_time;
                            }

                            if (torch_time_max_mkl < torch_completion_time)
                            {
                                torch_time_max_mkl = torch_completion_time;
                            }

                            if (nw_time_min_mkl > nw_completion_time)
                            {
                                nw_time_min_mkl = nw_completion_time;
                            }

                            if (nw_time_max_mkl < nw_completion_time)
                            {
                                nw_time_max_mkl = nw_completion_time;
                            }
                            break;
                        case CU_RUNTIME:
                            torch_time_cuda += (float64_t) torch_completion_time / total_runs;
                            nw_time_cuda += (float64_t) nw_completion_time / total_runs;

                            if (torch_time_min_cuda > torch_completion_time)
                            {
                                torch_time_min_cuda = torch_completion_time;
                            }

                            if (torch_time_max_cuda < torch_completion_time)
                            {
                                torch_time_max_cuda = torch_completion_time;
                            }

                            if (nw_time_min_cuda > nw_completion_time)
                            {
                                nw_time_min_cuda = nw_completion_time;
                            }

                            if (nw_time_max_cuda < nw_completion_time)
                            {
                                nw_time_max_cuda = nw_completion_time;
                            }
                            break;
                        default:
                        ck_abort_msg("unknown runtime.");
                    }
                }

            }
        }

        torch_time_arr_mkl[k] = torch_time_mkl;

        torch_time_arr_cuda[k] = torch_time_cuda;

        nw_time_arr_openblas[k] = nw_time_openblas;

        nw_time_arr_mkl[k] = nw_time_mkl;

        nw_time_arr_cuda[k] = nw_time_cuda;

        x[k] = n;
    }

    plot_heuristics("Torch MKL Completion Time - " + op_name,
            "buffer_" + op_name + "_torch_mkl_time.png", x, CASES,
            torch_time_arr_mkl, CASES);

    plot_heuristics("Torch CUDA Completion Time - " + op_name,
            "buffer_" + op_name + "_torch_cuda_time.png", x, CASES,
            torch_time_arr_cuda, CASES);

    plot_heuristics("NW OpenBLAS Completion Time - " + op_name,
            "buffer_" + op_name + "_nw_openblas_time.png", x, CASES,
            nw_time_arr_openblas, CASES);

    plot_heuristics("NW MKL Completion Time - " + op_name,
            "buffer_" + op_name + "_nw_mkl_time.png", x, CASES,
            nw_time_arr_mkl, CASES);

    plot_heuristics("NW CUDA Completion Time - " + op_name,
            "buffer_" + op_name + "_nw_cuda_time.png", x, CASES,
            nw_time_arr_cuda, CASES);
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
                    torch::Tensor expected_tensor = torch_op(torch_tensors[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    returned_tensors[i][j][k][z] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                    nw_start = get_time_nanoseconds();
                    error = nw_op(tensors[i][j][k][z]->buffer, returned_tensors[i][j][k][z]->buffer);
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
    performance_test("Exponential", AS_LAMBDA(torch::exp),
            AS_LAMBDA(runtime_exponential));
}
END_TEST

START_TEST(test_logarithm_computational_performance)
{
    performance_test("Logarithm", AS_LAMBDA(torch::log),
            AS_LAMBDA(runtime_logarithm));
}
END_TEST

START_TEST(test_sine_computational_performance)
{
    performance_test("Sine", AS_LAMBDA(torch::sin), AS_LAMBDA(runtime_sine));
}
END_TEST

START_TEST(test_cosine_computational_performance)
{
    performance_test("Cosine", AS_LAMBDA(torch::cos),
            AS_LAMBDA(runtime_cosine));
}
END_TEST

START_TEST(test_square_root_computational_performance)
{
    performance_test("Square_Root", AS_LAMBDA(torch::sqrt),
            AS_LAMBDA(runtime_square_root));
}
END_TEST

START_TEST(test_reciprocal_computational_performance)
{
    printf("---------------------   Reciprocal   ---------------------\n");
    performance_test(AS_LAMBDA(torch::reciprocal), AS_LAMBDA(runtime_reciprocal),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
}
END_TEST

START_TEST(test_contiguous_computational_performance)
{
    performance_test("Contiguous", AS_MEMBER_LAMBDA(torch::Tensor::contiguous),
            AS_LAMBDA(runtime_contiguous));
}
END_TEST

START_TEST(test_negation_computational_performance)
{
    performance_test("Negation", AS_LAMBDA(torch::neg),
            AS_LAMBDA(runtime_negation));
}
END_TEST

START_TEST(test_rectified_linear_computational_performance)
{
    printf("------------------   Rectified Linear   ------------------\n");
    performance_test(AS_LAMBDA(torch::relu), AS_LAMBDA(runtime_rectified_linear),
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
