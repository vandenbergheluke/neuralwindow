#include <check.h>
#include <tensor.h>
#include <errors.h>
#include <datatype.h>
#include <buffer.h>

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

uint64_t lower = 0;
uint64_t upper = 1000;
uint64_t mean = 0;
uint64_t std = 1;

uint64_t prev_nodes_num = 10;
uint64_t  gain = 1;

uint64_t fan_in = 3;
uint64_t fan_out = 4;

// Initialize shapes_x
const uint64_t shapes_x_0[] = {2};
uint64_t shapes_x_1[] = {3, 2};
uint64_t shapes_x_2[] = {2, 3, 4};
uint64_t shapes_x_3[] = {2, 3, 4, 5};
uint64_t shapes_x_4[] = {6, 5, 4, 3, 2};
uint64_t shapes_x_5[] = {3, 3, 3};
uint64_t shapes_x_6[] = {4, 4, 4};
uint64_t shapes_x_7[] = {2, 3, 4};
uint64_t shapes_x_8[] = {2, 3, 4, 5};
uint64_t shapes_x_9[] = {6, 5, 4, 3, 2};

// Initialize shapes_y
uint64_t shapes_y_0[] = {5};
uint64_t shapes_y_1[] = {2, 3};
uint64_t shapes_y_2[] = {2, 4, 3};
uint64_t shapes_y_3[] = {2, 3, 5, 4};
uint64_t shapes_y_4[] = {6, 5, 4, 2, 3};
uint64_t shapes_y_5[] = {3, 3, 3};
uint64_t shapes_y_6[] = {4, 4, 4};
uint64_t shapes_y_7[] = {2, 4, 3};
uint64_t shapes_y_8[] = {2, 3, 5, 4};
uint64_t shapes_y_9[] = {6, 5, 4, 2, 3};

// Initialize stride_x
uint64_t stride_x_0[] = {1};
uint64_t stride_x_1[] = {2, 1};
uint64_t stride_x_2[] = {6, 4, 1};
uint64_t stride_x_3[] = {120, 20, 5, 1};
uint64_t stride_x_4[] = {120, 24, 6, 2, 1};
uint64_t stride_x_5[] = {9, 3, 1};
uint64_t stride_x_6[] = {16, 4, 1};
uint64_t stride_x_7[] = {12, 4, 1};
uint64_t stride_x_8[] = {120, 20, 5, 1};
uint64_t stride_x_9[] = {120, 24, 6, 2, 1};

// Initialize stride_y
uint64_t stride_y_0[] = {1};
uint64_t stride_y_1[] = {3, 1};
uint64_t stride_y_2[] = {12, 3, 1};
uint64_t stride_y_3[] = {60, 20, 4, 1};
uint64_t stride_y_4[] = {120, 24, 6, 3, 1};
uint64_t stride_y_5[] = {9, 3, 1};
uint64_t stride_y_6[] = {16, 4, 1};
uint64_t stride_y_7[] = {12, 3, 1};
uint64_t stride_y_8[] = {60, 20, 4, 1};
uint64_t stride_y_9[] = {120, 24, 6, 3, 1};

void setup(void)
{
    error = NULL;
    //tensor_t **t = &x_tensors[0];
    //const uint64_t *sh = shapes_x_0;
    //const uint64_t *st = stride_x_0;
    /*
    error = tensor_create_zeroes(&x_tensors[0], shapes_x_0, 1, stride_x_0, 0, MKL_RUNTIME, FLOAT64, false, false);
    if (error)
   {
      error_print(error);
      error_destroy(error); 
   }
    
    
    error = tensor_create_ones(&x_tensors[1], shapes_x_1, 2, stride_x_1, 0, MKL_RUNTIME, FLOAT64, false, false);
    
    
    error = tensor_create_uniform(&x_tensors[2], shapes_x_2, 3, stride_x_2, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) lower, (void *) upper);
    
    
    error = tensor_create_normal(&x_tensors[3], shapes_x_3, 4, stride_x_3, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) mean, (void *) std);


    error = tensor_create_kaiming_uniform(&x_tensors[4], shapes_x_4, 5, stride_x_4, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) gain, (void *) prev_nodes_num);
    
    
    error = tensor_create_kaiming_normal(&x_tensors[5], shapes_x_5, 3, stride_x_5, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) gain, (void *) prev_nodes_num);

    
    error = tensor_create_glorot_uniform(&x_tensors[6], shapes_x_6, 3, stride_x_6, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) gain, (void *) fan_in, (void *) fan_out);
    
    
    error = tensor_create_glorot_normal(&x_tensors[7], shapes_x_7, 3, stride_x_7, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) gain, (void *) fan_in, (void *) fan_out);


    error = tensor_create_kaiming_normal(&x_tensors[8], shapes_x_8, 4, stride_x_8, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) gain, (void *) prev_nodes_num);
    
    
    error = tensor_create_glorot_uniform(&x_tensors[9], shapes_x_9, 5, stride_x_9, 0, MKL_RUNTIME, FLOAT64, false, false, (void *) gain, (void *) fan_in, (void *) fan_out);
    */
    ck_assert_ptr_null(error);
}

void teardown(void)
{
    for (int i = 0; i < 10; i++)
    {
        tensor_destroy(x_tensors[i]);
    }
}

START_TEST(test_graph)
{
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

    sr = srunner_create(graph_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}