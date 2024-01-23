#include <check.h>
#include <tensor.h>
#include <errors.h>
#include <datatype.h>
#include <buffer.h>
#include <graph.h>

#define CASES 10
#define CASES02 5

nw_error_t *error;

tensor_t *x_tensors[CASES];
tensor_t *x_unary_tensors[CASES];
tensor_t *y_tensors[CASES];
tensor_t *y_unary_tensors[CASES];

tensor_t *z_binary_tensors[CASES02];
tensor_t *z_reduction_tensors[CASES02];
tensor_t *z_structure_tensors[CASES02];

int64_t lower = 0;
int64_t upper = 1000;
int64_t mean = 0;
int64_t std = 1;

int64_t prev_nodes_num = 10;
int64_t  gain = 1;

int64_t fan_in = 3;
int64_t fan_out = 4;

// x tensors
tensor_t *x_0 = NULL;
tensor_t *x_1 = NULL;
tensor_t *x_2 = NULL;
tensor_t *x_3 = NULL;
tensor_t *x_4 = NULL;
tensor_t *x_5 = NULL;
tensor_t *x_6 = NULL;
tensor_t *x_7 = NULL;
tensor_t *x_8 = NULL;
tensor_t *x_9 = NULL;

// x output tensors
tensor_t *x_0_output = NULL;
tensor_t *x_1_output = NULL;
tensor_t *x_2_output = NULL;
tensor_t *x_3_output = NULL;
tensor_t *x_4_output = NULL;
tensor_t *x_5_output = NULL;
tensor_t *x_6_output = NULL;
tensor_t *x_7_output = NULL;
tensor_t *x_8_output = NULL;
tensor_t *x_9_output = NULL;

// y tensors
tensor_t *y_0 = NULL;
tensor_t *y_1 = NULL;
tensor_t *y_2 = NULL;
tensor_t *y_3 = NULL;
tensor_t *y_4 = NULL;
tensor_t *y_5 = NULL;
tensor_t *y_6 = NULL;
tensor_t *y_7 = NULL;
tensor_t *y_8 = NULL;
tensor_t *y_9 = NULL;

// y tensors
tensor_t *y_0_output = NULL;
tensor_t *y_1_output = NULL;
tensor_t *y_2_output = NULL;
tensor_t *y_3_output = NULL;
tensor_t *y_4_output = NULL;
tensor_t *y_5_output = NULL;
tensor_t *y_6_output = NULL;
tensor_t *y_7_output = NULL;
tensor_t *y_8_output = NULL;
tensor_t *y_9_output = NULL;

// z tensors 
tensor_t *z_0 = NULL;
tensor_t *z_1 = NULL;
tensor_t *z_2 = NULL;
tensor_t *z_3 = NULL;
tensor_t *z_4 = NULL;
tensor_t *z_5 = NULL;
tensor_t *z_6 = NULL;
tensor_t *z_7 = NULL;
tensor_t *z_8 = NULL;
tensor_t *z_9 = NULL;

// z tensors
tensor_t *z_0_output = NULL;
tensor_t *z_1_output = NULL;
tensor_t *z_2_output = NULL;
tensor_t *z_3_output = NULL;
tensor_t *z_4_output = NULL;
tensor_t *z_5_output = NULL;
tensor_t *z_6_output = NULL;
tensor_t *z_7_output = NULL;
tensor_t *z_8_output = NULL;
tensor_t *z_9_output = NULL;

// Initialize shapes_x
int64_t shapes_x_0[] = {5};
int64_t shapes_x_1[] = {3, 1};
int64_t shapes_x_2[] = {2, 3, 4};
int64_t shapes_x_3[] = {2, 3, 5, 1};
int64_t shapes_x_4[] = {6, 5, 4, 3, 2};
int64_t shapes_x_5[] = {3, 3, 3};
int64_t shapes_x_6[] = {4, 4, 4};
int64_t shapes_x_7[] = {2, 3, 4};
int64_t shapes_x_8[] = {2, 3, 5, 4};
int64_t shapes_x_9[] = {6, 5, 4, 3, 2};

const int64_t *shapes_x[10] = {
    shapes_x_0, shapes_x_1, shapes_x_2,
    shapes_x_3, shapes_x_4, shapes_x_5,
    shapes_x_6, shapes_x_7, shapes_x_8,
    shapes_x_9
};

int64_t ranks_x[10] = {1, 2, 3, 4, 5, 3, 3, 3, 4, 5};

// Initialize shapes_y
int64_t shapes_y_0[] = {5};
int64_t shapes_y_1[] = {3, 3};
int64_t shapes_y_2[] = {2, 3, 4};
int64_t shapes_y_3[] = {2, 3, 5, 4};
int64_t shapes_y_4[] = {6, 5, 4, 3, 1};
int64_t shapes_y_5[] = {3, 3, 3};
int64_t shapes_y_6[] = {4, 4, 4};
int64_t shapes_y_7[] = {2, 4, 3};
int64_t shapes_y_8[] = {2, 3, 5, 4};
int64_t shapes_y_9[] = {6, 5, 4, 2, 3};

const int64_t *shapes_y[10] = {
    shapes_y_0, shapes_y_1, shapes_y_2,
    shapes_y_3, shapes_y_4, shapes_y_5,
    shapes_y_6, shapes_y_7, shapes_y_8,
    shapes_y_9
};

int64_t ranks_y[10] = {1, 2, 3, 4, 5, 3, 3, 3, 4, 5};

void setup(void)
{
 
}

void teardown(void)
{
    tensor_destroy(x_0);
    tensor_destroy(x_1);
    tensor_destroy(x_2);
    tensor_destroy(x_3);
    tensor_destroy(x_4);
    tensor_destroy(x_5);
    tensor_destroy(x_6);
    tensor_destroy(x_7);
    tensor_destroy(x_8);
    tensor_destroy(x_9);

    tensor_destroy(x_0_output);
    tensor_destroy(x_1_output);
    tensor_destroy(x_2_output);
    tensor_destroy(x_3_output);
    tensor_destroy(x_4_output);
    tensor_destroy(x_5_output);
    tensor_destroy(x_6_output);
    tensor_destroy(x_7_output);
    tensor_destroy(x_8_output);
    tensor_destroy(x_9_output);

    tensor_destroy(y_0);
    tensor_destroy(y_1);
    tensor_destroy(y_2);
    tensor_destroy(y_3);
    tensor_destroy(y_4);
    tensor_destroy(y_5);
    tensor_destroy(y_6);
    tensor_destroy(y_7);
    tensor_destroy(y_8);
    tensor_destroy(y_9);

    tensor_destroy(y_0_output);
    tensor_destroy(y_1_output);
    tensor_destroy(y_2_output);
    tensor_destroy(y_3_output);
    tensor_destroy(y_4_output);
    tensor_destroy(y_5_output);
    tensor_destroy(y_6_output);
    tensor_destroy(y_7_output);
    tensor_destroy(y_8_output);
    tensor_destroy(y_9_output);

    tensor_destroy(z_0);
    tensor_destroy(z_1);
    tensor_destroy(z_2);
    tensor_destroy(z_3);
    tensor_destroy(z_4);
    tensor_destroy(z_5);
    tensor_destroy(z_6);
    tensor_destroy(z_7);
    tensor_destroy(z_8);
    tensor_destroy(z_9);

    tensor_destroy(z_0_output);
    tensor_destroy(z_1_output);
    tensor_destroy(z_2_output);
    tensor_destroy(z_3_output);
    tensor_destroy(z_4_output);
    tensor_destroy(z_5_output);
    tensor_destroy(z_6_output);
    tensor_destroy(z_7_output);
    tensor_destroy(z_8_output);
    tensor_destroy(z_9_output);

    destroy_graph();

}

START_TEST(test_graph)
{
   error = NULL;

    float gain = 2.0; 
    float fan = 1.0; 

    double gain_8 = 2.0; 
    double fan_8 = 1.0; 

    float gain_value_2 = 2.0;  
    float fan_value_2 = 3.0; 
    float  fan_out_value_2 = 5.0;

    int64_t lower_bound = 1; 
    int64_t upper_bound = 10; 

    int64_t mean = 0;
    int64_t std = 1;

    float gain_value = 2.0;  
    float fan_in_value = 4.0;  
    float fan_out_value = 6.0;

    // x0
    error = tensor_create_ones(&x_0, shapes_x_0, ranks_x[0], MKL_RUNTIME, FLOAT64, true, false);
    if (error)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(x_0, &x_0_output, false, false);
    ck_assert_ptr_null(error);

    // x1
    error = tensor_create_uniform(&x_1, shapes_x_1, ranks_x[1], MKL_RUNTIME, FLOAT64, false, false, &lower_bound, &upper_bound);
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(x_1, &x_1_output, false, false);
    ck_assert_ptr_null(error);

    // x2
    error = tensor_create_normal(&x_2, shapes_x_2, ranks_x[2], MKL_RUNTIME, FLOAT64, false, false, &mean, &std);
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(x_2, &x_2_output, false, false);
    ck_assert_ptr_null(error);

    // x3
    error = tensor_create_kaiming_uniform(&x_3, shapes_x_3, ranks_x[3], MKL_RUNTIME, FLOAT64, false, false, &gain, &fan);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(x_3, &x_3_output, false, false);
    ck_assert_ptr_null(error);

    // x4
    error = tensor_create_kaiming_normal(&x_4, shapes_x_4, ranks_x[4], MKL_RUNTIME, FLOAT64, false, false, &gain, &fan);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(x_4, &x_4_output, false, false);
    ck_assert_ptr_null(error);

    // x5
    error = tensor_create_glorot_uniform(&x_5, shapes_x_5, ranks_x[5], MKL_RUNTIME, FLOAT64, false, false, &gain_value, &fan_in_value, &fan_out_value);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(x_5, &x_5_output, false, false);
    ck_assert_ptr_null(error);

    // x6
    error = tensor_create_glorot_normal(&x_6, shapes_x_6, ranks_x[6], MKL_RUNTIME, FLOAT64, false, false, &gain_value, &fan_in_value, &fan_out_value);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(x_6, &x_6_output, false, false);
    ck_assert_ptr_null(error);

    // x7
    error = tensor_create_zeroes(&x_7, shapes_x_7, ranks_x[7], MKL_RUNTIME, FLOAT64, false, false);
    ck_assert_ptr_null(error);

    error = tensor_empty_like(x_7, &x_7_output, false, false);
    ck_assert_ptr_null(error);

    // x8
    error = tensor_create_kaiming_normal(&x_8, shapes_x_8, ranks_x[8], MKL_RUNTIME, FLOAT64, false, false, &gain_8, &fan_8);
    ck_assert_ptr_null(error);

    error = tensor_empty_like(x_8, &x_8_output, false, false);
    ck_assert_ptr_null(error);

    // x9
    error = tensor_create_glorot_uniform(&x_9, shapes_x_9, ranks_x[9], MKL_RUNTIME, FLOAT64, false, false, &gain_value_2, &fan_value_2, &fan_out_value_2);
    ck_assert_ptr_null(error);

    error = tensor_empty_like(x_9, &x_9_output, false, false);
    ck_assert_ptr_null(error);

    //y init

    // y0
    error = tensor_create_ones(&y_0, shapes_y_0, ranks_y[0], MKL_RUNTIME, FLOAT64, true, false);
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(y_0, &y_0_output, false, false);
    ck_assert_ptr_null(error);

    // y1
    error = tensor_create_uniform(&y_1, shapes_y_1, ranks_y[1], MKL_RUNTIME, FLOAT64, false, false, &lower_bound, &upper_bound);
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(y_1, &y_1_output, false, false);
    ck_assert_ptr_null(error);

    // y2
    error = tensor_create_normal(&y_2, shapes_y_2, ranks_y[2], MKL_RUNTIME, FLOAT64, false, false, &mean, &std);
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(y_2, &y_2_output, false, false);
    ck_assert_ptr_null(error);

    //y3
    error = tensor_create_kaiming_uniform(&y_3, shapes_y_3, ranks_y[3], MKL_RUNTIME, FLOAT64, false, false, &gain, &fan);
    ck_assert_ptr_null(error);

    error = tensor_zeroes_like(y_3, &y_3_output, false, false);
    ck_assert_ptr_null(error);

    // y4
    error = tensor_create_kaiming_normal(&y_4, shapes_y_4, ranks_y[4], MKL_RUNTIME, FLOAT64, false, false, &gain, &fan);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(y_4, &y_4_output, false, false);
    ck_assert_ptr_null(error);

    // y5
    error = tensor_create_glorot_uniform(&y_5, shapes_y_5, ranks_y[5], MKL_RUNTIME, FLOAT64, false, false, &gain_value, &fan_in_value, &fan_out_value);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(y_5, &y_5_output, false, false);
    ck_assert_ptr_null(error);

    // y6
    error = tensor_create_glorot_normal(&y_6, shapes_y_6, ranks_y[6], MKL_RUNTIME, FLOAT64, false, false, &gain_value, &fan_in_value, &fan_out_value);
    ck_assert_ptr_null(error);

    error = tensor_ones_like(y_6, &y_6_output, false, false);
    ck_assert_ptr_null(error);

    // y7
    error = tensor_create_zeroes(&y_7, shapes_y_7, ranks_y[7], MKL_RUNTIME, FLOAT64, false, false);
    ck_assert_ptr_null(error);

    error = tensor_empty_like(y_7, &y_7_output, false, false);
    ck_assert_ptr_null(error);

    // y8
    error = tensor_create_kaiming_normal(&y_8, shapes_y_8, ranks_y[8], MKL_RUNTIME, FLOAT64, false, false, &gain_value_2, &fan_value_2);
    ck_assert_ptr_null(error);

    error = tensor_empty_like(y_8, &y_8_output, false, false);
    ck_assert_ptr_null(error);

    // y9
    error = tensor_create_glorot_uniform(&y_9, shapes_y_9, ranks_y[9], MKL_RUNTIME, FLOAT64, false, false, &gain_value_2, &fan_value_2, &fan_out_value_2);
    ck_assert_ptr_null(error);

    error = tensor_empty_like(y_9, &y_9_output, false, false);
    ck_assert_ptr_null(error);

    // apply unary op on tensors x
    error = tensor_logarithm(x_0, &x_0_output);
    if (error)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_ptr_null(error);

    error = tensor_sine(x_1, &x_1_output);
    ck_assert_ptr_null(error);

    error = tensor_cosine(x_2, &x_2_output);
    if (error)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_ptr_null(error);
   
    error = tensor_logarithm(x_3, &x_3_output);
    ck_assert_ptr_null(error);

    error = tensor_square_root(x_4, &x_4_output);
    ck_assert_ptr_null(error);

    error = tensor_reciprocal(x_5, &x_5_output);
    ck_assert_ptr_null(error);

    error = tensor_negation(x_6, &x_6_output);
    ck_assert_ptr_null(error);

    error = tensor_rectified_linear(x_7, &x_7_output);
    ck_assert_ptr_null(error);

    error = tensor_sigmoid(x_8, &x_8_output);
    ck_assert_ptr_null(error);

    error = tensor_rectified_linear(x_9, &x_9_output);
    ck_assert_ptr_null(error);

    // apply unary op on tensors y
    error = tensor_logarithm(y_0, &y_0_output);
    ck_assert_ptr_null(error);

    error = tensor_sine(y_1, &y_1_output);
    ck_assert_ptr_null(error);

    error = tensor_cosine(y_2, &y_2_output);
    ck_assert_ptr_null(error);

    error = tensor_exponential(y_3, &y_3_output);
    ck_assert_ptr_null(error);

    error = tensor_square_root(y_4, &y_4_output);
    ck_assert_ptr_null(error);

    error = tensor_reciprocal(y_5, &y_5_output);
    ck_assert_ptr_null(error);

    error = tensor_negation(y_6, &y_6_output);
    ck_assert_ptr_null(error);

    error = tensor_rectified_linear(y_7, &y_7_output);
    ck_assert_ptr_null(error);

    error = tensor_sigmoid(y_8, &y_8_output);
    ck_assert_ptr_null(error);

    error = tensor_rectified_linear(y_9, &y_9_output);
    ck_assert_ptr_null(error);

    // apply binary op on tensors x and y
    error = tensor_addition(x_0_output, y_0_output, &z_0);
    ck_assert_ptr_null(error);

    error = tensor_subtraction(x_1_output, y_1_output, &z_1);
    ck_assert_ptr_null(error);

    error = tensor_division(x_2_output, y_2_output, &z_2);
    ck_assert_ptr_null(error);
    
    error = tensor_multiplication(x_3_output, y_3_output, &z_3);
    ck_assert_ptr_null(error);

    error = tensor_power(x_4_output, y_4_output, &z_4);
    ck_assert_ptr_null(error);

    error = tensor_matrix_multiplication(x_5_output, y_5_output, &z_5);
    ck_assert_ptr_null(error);

    error = tensor_compare_equal(x_6_output, y_6_output, &z_6);
    ck_assert_ptr_null(error);

    error = tensor_matrix_multiplication(x_7_output, y_7_output, &z_7);
    ck_assert_ptr_null(error);

    error = tensor_compare_greater(x_8_output, y_8_output, &z_8);
    ck_assert_ptr_null(error);

    error = tensor_matrix_multiplication(x_9_output, y_9_output, &z_9);
    ck_assert_ptr_null(error);

    // apply structure op on tensors z
    int64_t new_shape_z_0[] = {5, 5};
    error = tensor_expand(z_0, new_shape_z_0, 2, &z_0_output);
    ck_assert_ptr_null(error);

    int64_t new_shape_z_1[] = {9};
    error = tensor_reshape(z_1, &z_1_output, new_shape_z_1, 1);
    ck_assert_ptr_null(error);

    int64_t new_axis_z_2[] = {2, 0, 1};
    error = tensor_permute(z_2, &z_2_output, new_axis_z_2, 3);
    ck_assert_ptr_null(error);

    error = tensor_transpose(z_3, &z_3_output, 1, 2);
    ck_assert_ptr_null(error);

    error = tensor_transpose(z_4, &z_4_output, 2, 3);
    ck_assert_ptr_null(error);

    //Not Implemented Yet:
    //nw_error_t *tensor_slice(const tensor_t *x, tensor_t **y, int64_t *arguments, int64_t length);
    //int64_t new_args_z_4[] = {2, 2, 2, 1, 1, 1, 1, 0};
    //error = tensor_padding(z_4, &z_4_output, new_args_z_4, 30);


    // apply reduction op on tensors z
    int64_t new_axis_z_5[] = {};
    error = tensor_summation(z_5, &z_5_output, new_axis_z_5, 0, false);
    ck_assert_ptr_null(error);

    int64_t new_axis_z_6[] = {};
    error = tensor_maximum(z_6, &z_6_output, new_axis_z_6, 0, false);
    ck_assert_ptr_null(error);

    int64_t new_axis_z_7[] = {};
    error = tensor_mean(z_7, &z_7_output, new_axis_z_7, 0, false);
    ck_assert_ptr_null(error);

    error = tensor_softmax(z_8, &z_8_output, 2);
    ck_assert_ptr_null(error);

    error = tensor_logsoftmax(z_9, &z_9_output, 4);
    ck_assert_ptr_null(error);
}
END_TEST

Suite *graph_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Suite");
    tc_core = tcase_create("Case 1");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_graph);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    // Set the environment variable using putenv
    const char *env_variable = "GRAPH=1";
    if (putenv((char *)env_variable) != 0) {
        return EXIT_FAILURE;
    }

    sr = srunner_create(graph_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}