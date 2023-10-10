#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>

#include <mgl2/base_cf.h>
#include <mgl2/canvas_cf.h>
#include <mgl2/mgl_cf.h>
extern "C"
{
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <measure.h>
#include <tensor.h>

#include <check.h>
#include <math.h>
// TODO: Make this portable to windows
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
}
#include <test_helper.h>

// Take average over UT_MEASUREMENT_ITERS iterations.
#ifndef UT_MEASUREMENT_ITERS
#define UT_MEASUREMENT_ITERS 30
#endif

#define SAVE_DIR "img/tensor"

nw_error_t *error;

torch::Device device_cuda(torch::kCUDA);
torch::Device device_cpu(torch::kCPU);

static void mkdir_recurse(string_t s)
{
    char temp[PATH_MAX];
    char *c;
    
    snprintf(temp, sizeof(temp), "%s", s);

    c = temp;
    while (*c != '\0')
    {
        if (*c == '/')
        {
            *c = '\0';
            mkdir(temp, S_IRWXU);
            *c = '/';
        }
        ++c;
    }
    mkdir(temp, S_IRWXU);
}

void setup(void)
{
    struct stat st_result = {0};
    if (stat(SAVE_DIR, &st_result) == -1)
    {
        mkdir_recurse(SAVE_DIR);
    }

    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
    }
}

void teardown(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_destroy_context((runtime_t) i);
    }
    error_print(error);
    error_destroy(error);
}

//TODO: Make x axis ticks at multiples of 8.
void plot_heuristics(std::string t, std::string save_path,
        std::string x_str, float64_t* x, int x_n,
        std::string y_str, float64_t* y, int y_n,
        float64_t y_min, float64_t y_max)
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
    mgl_set_range_val(graph, 'y', y_min, y_max);
    mgl_axis(graph, "xy", "", "");
    // |    long dashed line
    // h    grey
    mgl_axis_grid(graph, "xy", "|h", "");
    mgl_label(graph, 'x', x_str.c_str(), 0, "");
    mgl_label(graph, 'y', y_str.c_str(), 0, "");
    mgl_box(graph);
    // u    blue purple
    // #.   circle-dot marker
    mgl_plot_xy(graph, x_mgl, y_mgl, "2u#.", "");

    mgl_write_png(graph, save_path.c_str(), "w");

    mgl_delete_graph(graph);
}

void plot_heuristics(std::string t, std::string save_path,
        std::string x_str, float64_t* x, int x_n,
        std::string y_str, float64_t* y1, int y1_n, std::string plt1,
        float64_t* y2, int y2_n, std::string plt2,
        float64_t y_min, float64_t y_max)
{
    HMGL graph = mgl_create_graph(800,400);

    HMDT x_mgl = mgl_create_data();
    HMDT y1_mgl = mgl_create_data();
    HMDT y2_mgl = mgl_create_data();

    mgl_data_set_double(x_mgl, x, x_n, 1, 1);
    mgl_data_set_double(y1_mgl, y1, y1_n, 1, 1);
    mgl_data_set_double(y2_mgl, y2, y2_n, 1, 1);

    // L    dark green blue
    // P    dark purple
    // #.   circle-dot marker
    mgl_add_legend(graph, plt1.c_str(), "2L#.");
    mgl_add_legend(graph, plt2.c_str(), "2P#.");

    // Colour values range from 0 to 1
    mgl_fill_background(graph, 1, 1, 1, 1);

    //mgl_subplot(graph, 3, 3, 4, "");
    mgl_inplot(graph, 0, 1, 0, 1);
    mgl_title(graph, t.c_str(), "", 5);
    mgl_set_range_dat(graph, 'x', x_mgl, 0);
    //mgl_set_range_dat(graph, 'y', y1_mgl, 0);
    mgl_set_range_val(graph, 'y', y_min, y_max);
    mgl_axis(graph, "xy", "", "");
    // |    long dashed line
    // h    grey
    mgl_axis_grid(graph, "xy", "|h", "");
    mgl_label(graph, 'x', x_str.c_str(), 0, "");
    mgl_label(graph, 'y', y_str.c_str(), 0, "");
    mgl_box(graph);
    // L    dark green blue
    // P    dark purple
    // #.   circle-dot marker
    mgl_plot_xy(graph, x_mgl, y1_mgl, "2L#.", "");
    mgl_plot_xy(graph, x_mgl, y2_mgl, "2P#.", "");

    // 0    absolute position of legend
    // NULL font style
    // #    draw box around legend
    mgl_legend(graph, 1, NULL, "#");

    mgl_write_png(graph, save_path.c_str(), "w");

    mgl_delete_graph(graph);
}

void performance_test(std::string op_name, datatype_t datatype,
        int rank, int max_shape_exp8,
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> torch_op,
        std::function<nw_error_t *(tensor_t *, tensor_t *, tensor_t **)> nw_op)
{
    ck_assert(0 < rank && rank <= 5);
    // Limited to 7 because of dim values stored as 64 bit signed int.
    ck_assert(0 < max_shape_exp8 && max_shape_exp8 < 8);

    int num_cases = (7 * max_shape_exp8) + 1;

    float64_t torch_time_arr_mkl[num_cases];
    //float64_t torch_time_arr_cuda[num_cases];

    float64_t nw_time_arr_mkl[num_cases];
    float64_t nw_time_arr_openblas[num_cases];
    float64_t nw_time_arr_cuda[num_cases];

    // Minimum time (for range calculations)
    float64_t torch_time_min_mkl = DBL_MAX;
    //float64_t torch_time_min_cuda = DBL_MAX;

    float64_t nw_time_min_mkl = DBL_MAX;
    float64_t nw_time_min_openblas = DBL_MAX;
    float64_t nw_time_min_cuda = DBL_MAX;

    // Maximum time (for range calculations)
    float64_t torch_time_max_mkl = DBL_MIN;
    //float64_t torch_time_max_cuda = DBL_MIN;

    float64_t nw_time_max_mkl = DBL_MIN;
    float64_t nw_time_max_openblas = DBL_MIN;
    float64_t nw_time_max_cuda = DBL_MIN;

    // Min and max between torch/nw.
    float64_t mkl_time_min;
    float64_t mkl_time_max;

    float64_t widths[num_cases];

    std::string op_name_dir = op_name;
    std::string op_save_dir;
    std::string rank_str = std::to_string(rank);

    for (int x = 0; x <= max_shape_exp8; ++x)
    {
        int y = 0;
        do
        {
            float64_t torch_time_mkl = 0;
            //float64_t torch_time_cuda = 0;
            float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;

            int64_t n = (y + 1) * pow(8, x);

            std::vector<int64_t> shape;

            for (int i = 0; i < rank; ++i)
            {
                shape.push_back(n);
            }

            for (int i = 0; i < RUNTIMES; ++i)
            {
                // Take average time of UT_MEASUREMENT_ITERS iterations for
                // each runtime.
                for (int j = 0; j < UT_MEASUREMENT_ITERS; ++j)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;

                    torch::Tensor torch_tensor_x;
                    torch::Tensor torch_tensor_y;

                    tensor_t *tensor_x;
                    tensor_t *tensor_y;
                    tensor_t *returned_tensor;

                    // We're using CPU pytorch because we use an unsupported
                    // version of CUDA... CUDA tests are disabled right now.
                    switch (datatype)
                    {
                    case FLOAT32:
                        torch_tensor_x = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        torch_tensor_y = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    case FLOAT64:
                        torch_tensor_x = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        torch_tensor_y = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }

                    tensor_x = torch_to_tensor(torch_tensor_x, (runtime_t) i, datatype);
                    tensor_y = torch_to_tensor(torch_tensor_y, (runtime_t) i, datatype);

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch_op(torch_tensor_x, torch_tensor_y);
                    torch_end = get_time_nanoseconds();

                    returned_tensor = torch_to_tensor(expected_tensor, (runtime_t) i, datatype);

                    nw_start = get_time_nanoseconds();
                    error = nw_op(tensor_x, tensor_y, &returned_tensor);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    tensor_destroy(tensor_x);
                    tensor_destroy(tensor_y);
                    tensor_destroy(returned_tensor);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    switch ((runtime_t) i)
                    {
                        case OPENBLAS_RUNTIME:
                            // Pytorch uses MKL on CPU

                            nw_time_openblas += (float64_t) nw_completion_time / UT_MEASUREMENT_ITERS;

                            break;
                        case MKL_RUNTIME:
                            // Torch MKL gets double the runs as a biproduct of
                            // how the tests are setup.

                            torch_time_mkl += (float64_t) torch_completion_time / (2 * UT_MEASUREMENT_ITERS);
                            nw_time_mkl += (float64_t) nw_completion_time / UT_MEASUREMENT_ITERS;

                            break;
                        case CU_RUNTIME:
                            //torch_time_cuda += (float64_t) torch_completion_time / UT_MEASUREMENT_ITERS;
                            nw_time_cuda += (float64_t) nw_completion_time / UT_MEASUREMENT_ITERS;

                            break;
                        default:
                        ck_abort_msg("unknown runtime.");
                    }
                }
            }

            torch_time_arr_mkl[(x * 7) + y] = torch_time_mkl;
            torch_time_min_mkl = std::min(torch_time_min_mkl, torch_time_mkl);
            torch_time_max_mkl = std::max(torch_time_max_mkl, torch_time_mkl);

            /*
            torch_time_arr_cuda[(x * 7) + y] = torch_time_cuda;
            torch_time_min_cuda = std::min(torch_time_min_cuda, torch_time_cuda);
            torch_time_max_cuda = std::max(torch_time_max_cuda, torch_time_cuda);
            */

            nw_time_arr_openblas[(x * 7) + y] = nw_time_openblas;
            nw_time_min_openblas = std::min(nw_time_min_openblas, nw_time_openblas);
            nw_time_max_openblas = std::max(nw_time_max_openblas, nw_time_openblas);

            nw_time_arr_mkl[(x * 7) + y] = nw_time_mkl;
            nw_time_min_mkl = std::min(nw_time_min_mkl, nw_time_mkl);
            nw_time_max_mkl = std::max(nw_time_max_mkl, nw_time_mkl);

            nw_time_arr_cuda[(x * 7) + y] = nw_time_cuda;
            nw_time_min_cuda = std::min(nw_time_min_cuda, nw_time_cuda);
            nw_time_max_cuda = std::max(nw_time_max_cuda, nw_time_cuda);

            widths[(x * 7) + y] = (float64_t) n;

            ++y;
        } while ((y < 7) && (x != max_shape_exp8));
    }

    std::for_each(op_name_dir.begin(), op_name_dir.end(), [] (char &c) {
            if (isupper(c))
            {
                c = tolower(c);
            }
            else if (isspace(c))
            {
                c = '_';
            }});

    op_save_dir = std::string(SAVE_DIR) + "/" + op_name_dir;
    struct stat st_result = {0};
    if (stat(op_save_dir.c_str(), &st_result) == -1)
    {
        mkdir(op_save_dir.c_str(), S_IRWXU);
    }

    /*
    plot_heuristics("Torch CUDA Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/torch_cuda_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", torch_time_arr_cuda, num_cases,
            torch_time_min_cuda, torch_time_max_cuda);
    */

    plot_heuristics("NW OpenBLAS Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/nw_openblas_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", nw_time_arr_openblas, num_cases,
            nw_time_min_openblas, nw_time_max_openblas);

    mkl_time_min = std::min(torch_time_min_mkl, nw_time_min_mkl);
    mkl_time_max = std::max(torch_time_max_mkl, nw_time_max_mkl);
    plot_heuristics("MKL Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/mkl_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", nw_time_arr_mkl, num_cases, "NeuralWindow",
            torch_time_arr_mkl, num_cases, "PyTorch",
            mkl_time_min, mkl_time_max);

    plot_heuristics("NW CUDA Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/nw_cuda_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", nw_time_arr_cuda, num_cases,
            nw_time_min_cuda, nw_time_max_cuda);
}

void performance_test(std::string op_name, datatype_t datatype,
        int rank, int max_shape_exp8,
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> torch_op,
        std::function<nw_error_t *(tensor_t *, tensor_t *, tensor_t **)> nw_op,
        std::function<uint64_t(uint64_t)> flop_calc)
{
    ck_assert(0 < rank && rank <= 5);
    // Limited to 7 because of dim values stored as 64 bit signed int.
    ck_assert(0 < max_shape_exp8 && max_shape_exp8 < 8);

    int num_cases = (7 * max_shape_exp8) + 1;

    float64_t torch_time_arr_mkl[num_cases];
    //float64_t torch_time_arr_cuda[num_cases];

    float64_t torch_flops_arr_mkl[num_cases];
    //float64_t torch_flops_arr_cuda[num_cases];

    float64_t nw_time_arr_mkl[num_cases];
    float64_t nw_time_arr_openblas[num_cases];
    float64_t nw_time_arr_cuda[num_cases];

    float64_t nw_flops_arr_mkl[num_cases];
    float64_t nw_flops_arr_openblas[num_cases];
    float64_t nw_flops_arr_cuda[num_cases];

    // Minimum time/throughput (for range calculations)
    float64_t torch_time_min_mkl = DBL_MAX;
    //float64_t torch_time_min_cuda = DBL_MAX;

    float64_t torch_flops_min_mkl = DBL_MAX;
    //float64_t torch_flops_min_cuda = DBL_MAX;

    float64_t nw_time_min_mkl = DBL_MAX;
    float64_t nw_time_min_openblas = DBL_MAX;
    float64_t nw_time_min_cuda = DBL_MAX;

    float64_t nw_flops_min_mkl = DBL_MAX;
    float64_t nw_flops_min_openblas = DBL_MAX;
    float64_t nw_flops_min_cuda = DBL_MAX;

    // Maximum time/throughput (for range calculations)
    float64_t torch_time_max_mkl = DBL_MIN;
    //float64_t torch_time_max_cuda = DBL_MIN;

    float64_t torch_flops_max_mkl = DBL_MIN;
    //float64_t torch_flops_max_cuda = DBL_MIN;

    float64_t nw_time_max_mkl = DBL_MIN;
    float64_t nw_time_max_openblas = DBL_MIN;
    float64_t nw_time_max_cuda = DBL_MIN;

    float64_t nw_flops_max_mkl = DBL_MIN;
    float64_t nw_flops_max_openblas = DBL_MIN;
    float64_t nw_flops_max_cuda = DBL_MIN;

    // Min and max between torch/nw.
    float64_t mkl_time_min;
    float64_t mkl_time_max;

    float64_t mkl_flops_min;
    float64_t mkl_flops_max;

    float64_t widths[num_cases];

    std::string op_name_dir = op_name;
    std::string op_save_dir;
    std::string rank_str = std::to_string(rank);

    for (int x = 0; x <= max_shape_exp8; ++x)
    {
        int y = 0;
        do
        {
            float64_t torch_time_mkl = 0;
            float64_t torch_flops_mkl = 0;
            //float64_t torch_time_cuda = 0;
            //float64_t torch_flops_cuda = 0;
            float64_t nw_time_mkl = 0, nw_time_openblas = 0, nw_time_cuda = 0;
            float64_t nw_flops_mkl = 0, nw_flops_openblas = 0, nw_flops_cuda = 0;

            uint64_t n = (y + 1) * pow(8, x);
            uint64_t num_flop = flop_calc(n);

            std::vector<int64_t> shape;

            for (int i = 0; i < rank; ++i)
            {
                shape.push_back(n);
            }

            for (int i = 0; i < RUNTIMES; ++i)
            {
                // Take average time of UT_MEASUREMENT_ITERS iterations for
                // each runtime.
                for (int j = 0; j < UT_MEASUREMENT_ITERS; ++j)
                {
                    uint64_t torch_start, torch_end;
                    uint64_t torch_completion_time;
                    uint64_t nw_start, nw_end;
                    uint64_t nw_completion_time;

                    torch::Tensor torch_tensor_x;
                    torch::Tensor torch_tensor_y;

                    tensor_t *tensor_x;
                    tensor_t *tensor_y;
                    tensor_t *returned_tensor;

                    // We're using CPU pytorch because we use an unsupported
                    // version of CUDA... CUDA tests are disabled right now.
                    switch (datatype)
                    {
                    case FLOAT32:
                        torch_tensor_x = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        torch_tensor_y = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    case FLOAT64:
                        torch_tensor_x = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        torch_tensor_y = torch::randn(shape,
                                torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                // .device(((runtime_t) i == CU_RUNTIME) ? device_cuda : device_cpu)
                                );
                        break;
                    default:
                        ck_abort_msg("unknown datatype.");
                    }

                    tensor_x = torch_to_tensor(torch_tensor_x, (runtime_t) i, datatype);
                    tensor_y = torch_to_tensor(torch_tensor_y, (runtime_t) i, datatype);

                    torch_start = get_time_nanoseconds();
                    torch::Tensor expected_tensor = torch_op(torch_tensor_x, torch_tensor_y);
                    torch_end = get_time_nanoseconds();

                    returned_tensor = torch_to_tensor(expected_tensor, (runtime_t) i, datatype);

                    nw_start = get_time_nanoseconds();
                    error = nw_op(tensor_x, tensor_y, &returned_tensor);
                    nw_end = get_time_nanoseconds();
                    ck_assert_ptr_null(error);

                    tensor_destroy(tensor_x);
                    tensor_destroy(tensor_y);
                    tensor_destroy(returned_tensor);

                    torch_completion_time = torch_end - torch_start;
                    nw_completion_time = nw_end - nw_start;

                    switch ((runtime_t) i)
                    {
                        case OPENBLAS_RUNTIME:
                            // Pytorch uses MKL on CPU

                            nw_time_openblas += (float64_t) nw_completion_time / UT_MEASUREMENT_ITERS;
                            nw_flops_openblas += ((float64_t) num_flop * 1000000000) / ((float64_t) nw_completion_time * UT_MEASUREMENT_ITERS);

                            break;
                        case MKL_RUNTIME:
                            // Torch MKL gets double the runs as a biproduct of
                            // how the tests are setup.

                            torch_time_mkl += (float64_t) torch_completion_time / (2 * UT_MEASUREMENT_ITERS);
                            torch_flops_mkl += ((float64_t) num_flop * 1000000000) / ((float64_t) torch_completion_time * 2 * UT_MEASUREMENT_ITERS);
                            nw_time_mkl += (float64_t) nw_completion_time / UT_MEASUREMENT_ITERS;
                            nw_flops_mkl += ((float64_t) num_flop * 1000000000) / ((float64_t) nw_completion_time * UT_MEASUREMENT_ITERS);

                            break;
                        case CU_RUNTIME:
                            //torch_time_cuda += (float64_t) torch_completion_time / UT_MEASUREMENT_ITERS;
                            //torch_flops_cuda += ((float64_t) num_flop * 1000000000) / ((float64_t) torch_completion_time * UT_MEASUREMENT_ITERS);
                            nw_time_cuda += (float64_t) nw_completion_time / UT_MEASUREMENT_ITERS;
                            nw_flops_cuda += ((float64_t) num_flop * 1000000000) / ((float64_t) nw_completion_time * UT_MEASUREMENT_ITERS);

                            break;
                        default:
                        ck_abort_msg("unknown runtime.");
                    }
                }
            }

            torch_time_arr_mkl[(x * 7) + y] = torch_time_mkl;
            torch_time_min_mkl = std::min(torch_time_min_mkl, torch_time_mkl);
            torch_time_max_mkl = std::max(torch_time_max_mkl, torch_time_mkl);

            torch_flops_arr_mkl[(x * 7) + y] = torch_flops_mkl;
            torch_flops_min_mkl = std::min(torch_flops_min_mkl, torch_flops_mkl);
            torch_flops_max_mkl = std::max(torch_flops_max_mkl, torch_flops_mkl);

            //torch_time_arr_cuda[(x * 7) + y] = torch_time_cuda;
            //torch_time_min_cuda = std::min(torch_time_min_cuda, torch_time_cuda);
            //torch_time_max_cuda = std::max(torch_time_max_cuda, torch_time_cuda);

            //torch_flops_arr_cuda[(x * 7) + y] = torch_flops_cuda;
            //torch_flops_min_cuda = std::min(torch_flops_min_cuda, torch_flops_cuda);
            //torch_flops_max_cuda = std::max(torch_flops_max_cuda, torch_flops_cuda);

            nw_time_arr_openblas[(x * 7) + y] = nw_time_openblas;
            nw_time_min_openblas = std::min(nw_time_min_openblas, nw_time_openblas);
            nw_time_max_openblas = std::max(nw_time_max_openblas, nw_time_openblas);

            nw_flops_arr_openblas[(x * 7) + y] = nw_flops_openblas;
            nw_flops_min_openblas = std::min(nw_flops_min_openblas, nw_flops_openblas);
            nw_flops_max_openblas = std::max(nw_flops_max_openblas, nw_flops_openblas);

            nw_time_arr_mkl[(x * 7) + y] = nw_time_mkl;
            nw_time_min_mkl = std::min(nw_time_min_mkl, nw_time_mkl);
            nw_time_max_mkl = std::max(nw_time_max_mkl, nw_time_mkl);

            nw_flops_arr_mkl[(x * 7) + y] = nw_flops_mkl;
            nw_flops_min_mkl = std::min(nw_flops_min_mkl, nw_flops_mkl);
            nw_flops_max_mkl = std::max(nw_flops_max_mkl, nw_flops_mkl);

            nw_time_arr_cuda[(x * 7) + y] = nw_time_cuda;
            nw_time_min_cuda = std::min(nw_time_min_cuda, nw_time_cuda);
            nw_time_max_cuda = std::max(nw_time_max_cuda, nw_time_cuda);

            nw_flops_arr_cuda[(x * 7) + y] = nw_flops_cuda;
            nw_flops_min_cuda = std::min(nw_flops_min_cuda, nw_flops_cuda);
            nw_flops_max_cuda = std::max(nw_flops_max_cuda, nw_flops_cuda);

            widths[(x * 7) + y] = (float64_t) n;

            ++y;
        } while ((y < 7) && (x != max_shape_exp8));
    }

    std::for_each(op_name_dir.begin(), op_name_dir.end(), [] (char &c) {
            if (isupper(c))
            {
                c = tolower(c);
            }
            else if (isspace(c))
            {
                c = '_';
            }});

    op_save_dir = std::string(SAVE_DIR) + "/" + op_name_dir;
    struct stat st_result = {0};
    if (stat(op_save_dir.c_str(), &st_result) == -1)
    {
        mkdir(op_save_dir.c_str(), S_IRWXU);
    }

    //plot_heuristics("Torch CUDA Completion Time - R^" + rank_str + " " + op_name,
    //        op_save_dir + "/torch_cuda_time_r" + rank_str + ".png",
    //        "Square Matrix Width", widths, num_cases,
    //        "Time (nsec)", torch_time_arr_cuda, num_cases,
    //        torch_time_min_cuda, torch_time_max_cuda);

    plot_heuristics("NW OpenBLAS Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/nw_openblas_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", nw_time_arr_openblas, num_cases,
            nw_time_min_openblas, nw_time_max_openblas);

    mkl_time_min = std::min(torch_time_min_mkl, nw_time_min_mkl);
    mkl_time_max = std::max(torch_time_max_mkl, nw_time_max_mkl);
    plot_heuristics("MKL Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/mkl_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", nw_time_arr_mkl, num_cases, "NeuralWindow",
            torch_time_arr_mkl, num_cases, "PyTorch",
            mkl_time_min, mkl_time_max);

    plot_heuristics("NW CUDA Completion Time - R^" + rank_str + " " + op_name,
            op_save_dir + "/nw_cuda_time_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Time (nsec)", nw_time_arr_cuda, num_cases,
            nw_time_min_cuda, nw_time_max_cuda);

    //plot_heuristics("Torch CUDA Throughput - R^" + rank_str + " " + op_name,
    //        op_save_dir + "/torch_cuda_flops_r" + rank_str + ".png",
    //        "Square Matrix Width", widths, num_cases,
    //        "Throughput (FLOPS)", torch_flops_arr_cuda, num_cases,
    //        torch_flops_min_cuda, torch_flops_max_cuda);

    plot_heuristics("NW OpenBLAS Completion Throughput - R^" + rank_str + " " + op_name,
            op_save_dir + "/nw_openblas_flops_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Throughput (FLOPS)", nw_flops_arr_openblas, num_cases,
            nw_flops_min_openblas, nw_flops_max_openblas);

    mkl_flops_min = std::min(torch_flops_min_mkl, nw_flops_min_mkl);
    mkl_flops_max = std::max(torch_flops_max_mkl, nw_flops_max_mkl);
    plot_heuristics("MKL Completion Throughput - R^" + rank_str + " " + op_name,
            op_save_dir + "/mkl_flops_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Throughput (FLOPS)", nw_flops_arr_mkl, num_cases, "NeuralWindow",
            torch_flops_arr_mkl, num_cases, "PyTorch",
            mkl_flops_min, mkl_flops_max);

    plot_heuristics("NW CUDA Completion Throughput - R^" + rank_str + " " + op_name,
            op_save_dir + "/nw_cuda_flops_r" + rank_str + ".png",
            "Square Matrix Width", widths, num_cases,
            "Throughput (FLOPS)", nw_flops_arr_cuda, num_cases,
            nw_flops_min_cuda, nw_flops_max_cuda);
}

START_TEST(test_addition_computational_performance)
{
    // Rank 1
    performance_test("Addition FLOAT32", FLOAT32, 1, 4,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return n; });
    performance_test("Addition FLOAT64", FLOAT64, 1, 4,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return n; });
    // Rank 2
    performance_test("Addition FLOAT32", FLOAT32, 2, 3,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    performance_test("Addition FLOAT64", FLOAT64, 2, 3,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    // Rank 3
    performance_test("Addition FLOAT32", FLOAT32, 3, 2,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    performance_test("Addition FLOAT64", FLOAT64, 3, 2,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    // Rank 4
    performance_test("Addition FLOAT32", FLOAT32, 4, 1,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    performance_test("Addition FLOAT64", FLOAT64, 4, 1,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    // Rank 5
    performance_test("Addition FLOAT32", FLOAT32, 5, 1,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
    performance_test("Addition FLOAT64", FLOAT64, 5, 1,
            AS_LAMBDA(torch::add), AS_LAMBDA(tensor_addition),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
}
END_TEST

START_TEST(test_subtraction_computational_performance)
{
    // Rank 1
    performance_test("Subtraction FLOAT32", FLOAT32, 1, 4,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return n; });
    performance_test("Subtraction FLOAT64", FLOAT64, 1, 4,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return n; });
    // Rank 2
    performance_test("Subtraction FLOAT32", FLOAT32, 2, 3,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    performance_test("Subtraction FLOAT64", FLOAT64, 2, 3,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    // Rank 3
    performance_test("Subtraction FLOAT32", FLOAT32, 3, 2,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    performance_test("Subtraction FLOAT64", FLOAT64, 3, 2,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    // Rank 4
    performance_test("Subtraction FLOAT32", FLOAT32, 4, 1,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    performance_test("Subtraction FLOAT64", FLOAT64, 4, 1,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    // Rank 5
    performance_test("Subtraction FLOAT32", FLOAT32, 5, 1,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
    performance_test("Subtraction FLOAT64", FLOAT64, 5, 1,
            AS_LAMBDA(torch::subtract), AS_LAMBDA(tensor_subtraction),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
}
END_TEST

START_TEST(test_multiplication_computational_performance)
{
    // Rank 1
    performance_test("Multiplication FLOAT32", FLOAT32, 1, 4,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return n; });
    performance_test("Multiplication FLOAT64", FLOAT64, 1, 4,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return n; });
    // Rank 2
    performance_test("Multiplication FLOAT32", FLOAT32, 2, 3,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    performance_test("Multiplication FLOAT64", FLOAT64, 2, 3,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    // Rank 3
    performance_test("Multiplication FLOAT32", FLOAT32, 3, 2,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    performance_test("Multiplication FLOAT64", FLOAT64, 3, 2,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    // Rank 4
    performance_test("Multiplication FLOAT32", FLOAT32, 4, 1,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    performance_test("Multiplication FLOAT64", FLOAT64, 4, 1,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    // Rank 5
    performance_test("Multiplication FLOAT32", FLOAT32, 5, 1,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
    performance_test("Multiplication FLOAT64", FLOAT64, 5, 1,
            AS_LAMBDA(torch::mul), AS_LAMBDA(tensor_multiplication),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
}
END_TEST

START_TEST(test_division_computational_performance)
{
    // Rank 1
    performance_test("Division FLOAT32", FLOAT32, 1, 4,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return n; });
    performance_test("Division FLOAT64", FLOAT64, 1, 4,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return n; });
    // Rank 2
    performance_test("Division FLOAT32", FLOAT32, 2, 3,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    performance_test("Division FLOAT64", FLOAT64, 2, 3,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 2); });
    // Rank 3
    performance_test("Division FLOAT32", FLOAT32, 3, 2,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    performance_test("Division FLOAT64", FLOAT64, 3, 2,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 3); });
    // Rank 4
    performance_test("Division FLOAT32", FLOAT32, 4, 1,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    performance_test("Division FLOAT64", FLOAT64, 4, 1,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 4); });
    // Rank 5
    performance_test("Division FLOAT32", FLOAT32, 5, 1,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
    performance_test("Division FLOAT64", FLOAT64, 5, 1,
            AS_LAMBDA(torch::div), AS_LAMBDA(tensor_division),
            [] (uint64_t n) -> uint64_t { return pow(n, 5); });
}
END_TEST

START_TEST(test_power_computational_performance)
{
    // Rank 1
    performance_test("Power FLOAT32", FLOAT32, 1, 4,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    performance_test("Power FLOAT64", FLOAT64, 1, 4,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    // Rank 2
    performance_test("Power FLOAT32", FLOAT32, 2, 3,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    performance_test("Power FLOAT64", FLOAT64, 2, 3,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    // Rank 3
    performance_test("Power FLOAT32", FLOAT32, 3, 2,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    performance_test("Power FLOAT64", FLOAT64, 3, 2,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    // Rank 4
    performance_test("Power FLOAT32", FLOAT32, 4, 1,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    performance_test("Power FLOAT64", FLOAT64, 4, 1,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    // Rank 5
    performance_test("Power FLOAT32", FLOAT32, 5, 1,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
    performance_test("Power FLOAT64", FLOAT64, 5, 1,
            AS_LAMBDA(torch::pow), AS_LAMBDA(tensor_power));
}
END_TEST

START_TEST(test_matrix_multiplication_computational_performance)
{
    // Rank 1 - NA
    // Rank 2
    performance_test("Matrix Multiplication FLOAT32", FLOAT32, 2, 3,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 3); });
    performance_test("Matrix Multiplication FLOAT64", FLOAT64, 2, 3,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 3); });
    // Rank 3
    performance_test("Matrix Multiplication FLOAT32", FLOAT32, 3, 2,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 4); });
    performance_test("Matrix Multiplication FLOAT64", FLOAT64, 3, 2,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 4); });
    // Rank 4
    performance_test("Matrix Multiplication FLOAT32", FLOAT32, 4, 1,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 5); });
    performance_test("Matrix Multiplication FLOAT64", FLOAT64, 4, 1,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 5); });
    // Rank 5
    performance_test("Matrix Multiplication FLOAT32", FLOAT32, 5, 1,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 6); });
    performance_test("Matrix Multiplication FLOAT64", FLOAT64, 5, 1,
            AS_LAMBDA(torch::matmul),
            AS_LAMBDA(tensor_matrix_multiplication),
            [] (uint64_t n) -> uint64_t { return 2 * pow(n, 6); });
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
