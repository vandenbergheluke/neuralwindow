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
#include <math.h>
}
#include <torch/torch.h>

#define CASES 4

#define MEASUREMENT_ITERS 15

nw_error_t *error;

buffer_t *buffers_x[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
buffer_t *buffers_y[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

view_t *views_x[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
view_t *views_y[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
view_t *returned_views[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

torch::Tensor tensors_x[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];
torch::Tensor tensors_y[RUNTIMES][DATATYPES][CASES][MEASUREMENT_ITERS];

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
                    buffers_x[i][j][k][z] = NULL;
                    buffers_y[i][j][k][z] = NULL;
                    returned_buffers[i][j][k][z] = NULL;

                    views_x[i][j][k][z] = NULL;
                    views_y[i][j][k][z] = NULL;
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
                        tensors_x[i][j][k][z] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                        tensors_y[i][j][k][z] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                        break;
                    case FLOAT64:
                        tensors_x[i][j][k][z] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                        tensors_y[i][j][k][z] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }

                    error = view_create(&views_x[i][j][k][z], 
                                        (uint64_t) tensors_x[i][j][k][z].storage_offset(),
                                        (uint64_t) tensors_x[i][j][k][z].ndimension(),
                                        (uint64_t *) tensors_x[i][j][k][z].sizes().data(),
                                        NULL);
                    ck_assert_ptr_null(error);
                    error = buffer_create(&buffers_x[i][j][k][z],
                                          (runtime_t) i,
                                          (datatype_t) j,
                                          views_x[i][j][k][z],
                                          (void *) tensors_x[i][j][k][z].data_ptr(),
                                          (uint64_t) tensors_x[i][j][k][z].numel(),
                                          true);
                    ck_assert_ptr_null(error);

                    error = view_create(&views_y[i][j][k][z], 
                                        (uint64_t) tensors_y[i][j][k][z].storage_offset(),
                                        (uint64_t) tensors_y[i][j][k][z].ndimension(),
                                        (uint64_t *) tensors_y[i][j][k][z].sizes().data(),
                                        NULL);
                    ck_assert_ptr_null(error);
                    error = buffer_create(&buffers_y[i][j][k][z],
                                          (runtime_t) i,
                                          (datatype_t) j,
                                          views_y[i][j][k][z],
                                          (void *) tensors_y[i][j][k][z].data_ptr(),
                                          (uint64_t) tensors_y[i][j][k][z].numel(),
                                          true);
                    ck_assert_ptr_null(error);

                    error = view_create(&returned_views[i][j][k][z],
                                        (uint64_t) tensors_x[i][j][k][z].storage_offset(),
                                        (uint64_t) tensors_x[i][j][k][z].ndimension(),
                                        (uint64_t *) tensors_x[i][j][k][z].sizes().data(),
                                        NULL);
                    ck_assert_ptr_null(error);
                    error = buffer_create(&returned_buffers[i][j][k][z],
                                          (runtime_t) i,
                                          (datatype_t) j,
                                          returned_views[i][j][k][z],
                                          NULL,
                                          (uint64_t) tensors_x[i][j][k][z].numel(),
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
                    buffer_destroy(buffers_x[i][j][k][z]);
                    buffer_destroy(buffers_y[i][j][k][z]);
                    buffer_destroy(returned_buffers[i][j][k][z]);
                }
            }
        }
    }
    error_print(error);
    error_destroy(error);
}

START_TEST(test_addition_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = pow(n, 2);
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::add(tensors_x[i][j][k][z], tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_addition(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
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

    printf("PyTorch addition performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW addition performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

START_TEST(test_subtraction_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = pow(n, 2);
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::subtract(tensors_x[i][j][k][z], tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_subtraction(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
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

    printf("PyTorch subtraction performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW subtraction performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

START_TEST(test_multiplication_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = pow(n, 2);
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::mul(tensors_x[i][j][k][z], tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_multiplication(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
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

    printf("PyTorch multiplication performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW multiplication performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

START_TEST(test_division_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = pow(n, 2);
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::div(tensors_x[i][j][k][z], tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_division(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
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

    printf("PyTorch division performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW division performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

START_TEST(test_power_computational_performance)
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
                    torch::Tensor expected_tensor = torch::pow(tensors_x[i][j][k][z], tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_power(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    nw_avg_perf += (float64_t) nw_completion_time / total_runs;
                }
            }
        }
    }

    printf("PyTorch power performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW power performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_compare_equal_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = pow(n, 2);
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::eq(tensors_x[i][j][k][z], tensors_y[i][j][k][z]).to(tensors_x[i][j][k][z].dtype());
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_compare_equal(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
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

    printf("PyTorch compare equal performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW compare equal performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

START_TEST(test_compare_greater_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    uint64_t num_flop = pow(n, 2);
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::gt(tensors_x[i][j][k][z], tensors_y[i][j][k][z]).to(tensors_x[i][j][k][z].dtype());
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_compare_greater(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
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

    printf("PyTorch compare greater performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW compare greater performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

START_TEST(test_matrix_multiplication_computational_performance)
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
                    uint64_t n = ((uint64_t *) tensors_x[i][j][k][z].sizes().data())[0];
                    // HPLinpack
                    float64_t num_flop = ((2.0 / 3.0) * (float64_t) pow(n, 3))
                                        + (2.0 * (float64_t) pow(n, 2));
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;
                    uint32_t total_runs = RUNTIMES * DATATYPES * CASES * MEASUREMENT_ITERS;

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch::matmul(tensors_x[i][j][k][z], tensors_y[i][j][k][z]);
                    torch_end = get_time_nanoseconds();

                    nw_start = get_time_nanoseconds();
                    error = runtime_matrix_multiplication(buffers_x[i][j][k][z], buffers_y[i][j][k][z], returned_buffers[i][j][k][z]);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    torch_avg_perf += (float64_t) torch_completion_time / total_runs;
                    torch_avg_flops += (num_flop / (float64_t) torch_completion_time) / total_runs;
                    nw_avg_perf += (float64_t) nw_completion_time / total_runs;
                    nw_avg_flops += ((float64_t) num_flop / (float64_t) nw_completion_time) / total_runs;
                }
            }
        }
    }

    printf("PyTorch matrix multiplication performance (nsec): %0.2lf\n", torch_avg_perf);
    printf("NW matrix multiplication performance (nsec): %0.2lf\n", nw_avg_perf);
    printf("Fraction (NW nsec/Pytorch nsec): %0.3lf\n", nw_avg_perf / torch_avg_perf);
    printf("PyTorch FLOPS: %0.2lf\n", torch_avg_flops);
    printf("NW FLOPS: %0.2lf\n\n", nw_avg_flops);
}
END_TEST

Suite *make_buffer_binary_perf_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Buffer Binary Performance Suite");

    // Unary Performance Operations
    tc_unary = tcase_create("Buffer Binary Case");
    tcase_add_checked_fixture(tc_unary, setup, teardown);
    tcase_add_test(tc_unary, test_addition_computational_performance);
    tcase_add_test(tc_unary, test_subtraction_computational_performance);
    tcase_add_test(tc_unary, test_multiplication_computational_performance);
    tcase_add_test(tc_unary, test_division_computational_performance);
    tcase_add_test(tc_unary, test_power_computational_performance);
    tcase_add_test(tc_unary, test_compare_equal_computational_performance);
    tcase_add_test(tc_unary, test_compare_greater_computational_performance);
    tcase_add_test(tc_unary, test_matrix_multiplication_computational_performance);

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_binary_perf_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
