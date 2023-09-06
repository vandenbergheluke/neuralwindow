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

#define UNARY_PERF_CASES 2

nw_error_t *unary_perf_error;

buffer_t *unary_perf_buffers[UNARY_PERF_CASES];
buffer_t *returned_unary_perf_buffers[UNARY_PERF_CASES];
buffer_t *expected_unary_perf_buffers[UNARY_PERF_CASES];

view_t *unary_perf_views[UNARY_PERF_CASES];
view_t *returned_unary_perf_views[UNARY_PERF_CASES];
view_t *expected_unary_perf_views[UNARY_PERF_CASES];

torch::Tensor unary_perf_tensors[UNARY_PERF_CASES];

void unary_perf_setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        unary_perf_buffers[i] = NULL;
        returned_unary_perf_buffers[i] = NULL;
        expected_unary_perf_buffers[i] = NULL;

        unary_perf_views[i] = NULL;
        returned_unary_perf_views[i] = NULL;
        expected_unary_perf_views[i] = NULL;
    }

    // Must be square matrix.
    std::vector<int64_t> shapes[UNARY_PERF_CASES] = {
        {4, 4},
        {4, 4},
    };
    
    runtime_t runtimes[UNARY_PERF_CASES] = {
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
    };

    datatype_t datatypes[UNARY_PERF_CASES] = {
        FLOAT32,
        FLOAT32,
    };

    torch::ScalarType torch_datatypes[UNARY_PERF_CASES] = {
        torch::kFloat32,
        torch::kFloat32,
    };

    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        unary_perf_tensors[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));

        unary_perf_error = view_create(&unary_views[i], 
                                  (uint64_t) unary_perf_tensors[i].storage_offset(),
                                  (uint64_t) unary_perf_tensors[i].ndimension(),
                                  (uint64_t *) unary_perf_tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_perf_error);
        unary_perf_error = buffer_create(&unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    unary_perf_views[i],
                                    (void *) unary_perf_tensors[i].data_ptr(),
                                    (uint64_t) unary_perf_tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_perf_error);

        unary_perf_error = view_create(&returned_unary_views[i],
                                  (uint64_t) unary_perf_tensors[i].storage_offset(),
                                  (uint64_t) unary_perf_tensors[i].ndimension(),
                                  (uint64_t *) unary_perf_tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_perf_error);
        unary_perf_error = buffer_create(&returned_unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    returned_unary_perf_views[i],
                                    NULL,
                                    (uint64_t) unary_perf_tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_perf_error);
    }
}

void unary_perf_teardown(void)
{
    for (int i = 0; i < UNARY_PERF_CASES; i++)
    {
        buffer_destroy(unary_perf_buffers[i]);
        buffer_destroy(returned_unary_perf_buffers[i]);
        buffer_destroy(expected_unary_perf_buffers[i]);
    }
    error_destroy(unary_perf_error);
}

START_TEST(test_exponential_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        for (int j = 0; j < MEASUREMENT_ITERS; ++j)
        {
            uint64_t torch_start, torch_end;
            uint64_t nw_start, nw_end;

            torch_start = get_time_nanoseconds();
            torch::Tensor expected_tensor = torch::exp(unary_perf_tensors[i]);
            torch_end = get_time_nanoseconds();

            nw_start = get_time_nanoseconds();
            unary_perf_error = runtime_exponential(unary_perf_buffers[i], returned_unary_perf_buffers[i]);
            nw_end = get_time_nanoseconds();
            ck_assert_ptr_null(unary_perf_error);

            torch_avg_perf += (float64_t) (torch_end - torch_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
            nw_avg_perf += (float64_t) (nw_end - nw_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
        }
    }

    printf("PyTorch exponential performance (nsec): %lf\n", torch_avg_perf);
    printf("NW exponential performance (nsec): %lf\n", nw_avg_perf);
    printf("Fraction (NW/PyTorch): %lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_logarithm_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        for (int j = 0; j < MEASUREMENT_ITERS; ++j)
        {
            uint64_t torch_start, torch_end;
            uint64_t nw_start, nw_end;

            torch_start = get_time_nanoseconds();
            torch::Tensor expected_tensor = torch::log(unary_perf_tensors[i]);
            torch_end = get_time_nanoseconds();

            nw_start = get_time_nanoseconds();
            unary_perf_error = runtime_logarithm(unary_perf_buffers[i], returned_unary_perf_buffers[i]);
            nw_end = get_time_nanoseconds();
            ck_assert_ptr_null(unary_perf_error);

            torch_avg_perf += (float64_t) (torch_end - torch_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
            nw_avg_perf += (float64_t) (nw_end - nw_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
        }
    }

    printf("PyTorch logarithm performance (nsec): %lf\n", torch_avg_perf);
    printf("NW logarithm performance (nsec): %lf\n", nw_avg_perf);
    printf("Fraction (NW/PyTorch): %lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_sine_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        for (int j = 0; j < MEASUREMENT_ITERS; ++j)
        {
            uint64_t torch_start, torch_end;
            uint64_t nw_start, nw_end;

            torch_start = get_time_nanoseconds();
            torch::Tensor expected_tensor = torch::sin(unary_perf_tensors[i]);
            torch_end = get_time_nanoseconds();

            nw_start = get_time_nanoseconds();
            unary_perf_error = runtime_sine(unary_perf_buffers[i], returned_unary_perf_buffers[i]);
            nw_end = get_time_nanoseconds();
            ck_assert_ptr_null(unary_perf_error);

            torch_avg_perf += (float64_t) (torch_end - torch_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
            nw_avg_perf += (float64_t) (nw_end - nw_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
        }
    }

    printf("PyTorch sine performance (nsec): %lf\n", torch_avg_perf);
    printf("NW sine performance (nsec): %lf\n", nw_avg_perf);
    printf("Fraction (NW/PyTorch): %lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_cosine_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        for (int j = 0; j < MEASUREMENT_ITERS; ++j)
        {
            uint64_t torch_start, torch_end;
            uint64_t nw_start, nw_end;

            torch_start = get_time_nanoseconds();
            torch::Tensor expected_tensor = torch::cos(unary_perf_tensors[i]);
            torch_end = get_time_nanoseconds();

            nw_start = get_time_nanoseconds();
            unary_perf_error = runtime_cosine(unary_perf_buffers[i], returned_unary_perf_buffers[i]);
            nw_end = get_time_nanoseconds();
            ck_assert_ptr_null(unary_perf_error);

            torch_avg_perf += (float64_t) (torch_end - torch_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
            nw_avg_perf += (float64_t) (nw_end - nw_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
        }
    }

    printf("PyTorch cosine performance (nsec): %lf\n", torch_avg_perf);
    printf("NW cosine performance (nsec): %lf\n", nw_avg_perf);
    printf("Fraction (NW/PyTorch): %lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_square_root_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t nw_avg_perf = 0;
    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        for (int j = 0; j < MEASUREMENT_ITERS; ++j)
        {
            uint64_t torch_start, torch_end;
            uint64_t nw_start, nw_end;

            torch_start = get_time_nanoseconds();
            torch::Tensor expected_tensor = torch::sqrt(unary_perf_tensors[i]);
            torch_end = get_time_nanoseconds();

            nw_start = get_time_nanoseconds();
            unary_perf_error = runtime_square_root(unary_perf_buffers[i], returned_unary_perf_buffers[i]);
            nw_end = get_time_nanoseconds();
            ck_assert_ptr_null(unary_perf_error);

            torch_avg_perf += (float64_t) (torch_end - torch_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
            nw_avg_perf += (float64_t) (nw_end - nw_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
        }
    }

    printf("PyTorch square root performance (nsec): %lf\n", torch_avg_perf);
    printf("NW square root performance (nsec): %lf\n", nw_avg_perf);
    printf("Fraction (NW/PyTorch): %lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST

START_TEST(test_reciprocal_computational_performance)
{
    float64_t torch_avg_perf = 0;
    float64_t torch_avg_flops = 0;
    float64_t nw_avg_perf = 0;
    float64_t nw_avg_flops = 0;
    for (int i = 0; i < UNARY_PERF_CASES; ++i)
    {
        for (int j = 0; j < MEASUREMENT_ITERS; ++j)
        {
            uint64_t n = ((uint64_t *) unary_perf_tensors[i].sizes().data())[0];
            uint64_t torch_start, torch_end;
            uint64_t nw_start, nw_end;

            torch_start = get_time_nanoseconds();
            torch::Tensor expected_tensor = torch::reciprocal(unary_perf_tensors[i]);
            torch_end = get_time_nanoseconds();

            nw_start = get_time_nanoseconds();
            unary_perf_error = runtime_reciprocal(unary_perf_buffers[i], returned_unary_perf_buffers[i]);
            nw_end = get_time_nanoseconds();
            ck_assert_ptr_null(unary_perf_error);

            torch_avg_perf += (float64_t) (torch_end - torch_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
            torch_avg_flops
            nw_avg_perf += (float64_t) (nw_end - nw_start) / (UNARY_PERF_CASES * MEASUREMENT_ITERS);
        }
    }

    printf("PyTorch reciprocal performance (nsec): %lf\n", torch_avg_perf);
    printf("NW reciprocal performance (nsec): %lf\n", nw_avg_perf);
    printf("Fraction (NW/PyTorch): %lf\n", nw_avg_perf / torch_avg_perf);
}
END_TEST
