#include <iostream>
extern "C"
{
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
}
#include <test_helper.h>


#define BINARY_ELEMENTWISE_CASES_0_0 9
#define BINARY_ELEMENTWISE_CASES_1_0 15
#define BINARY_ELEMENTWISE_CASES_2_0 25
#define BINARY_ELEMENTWISE_CASES_3_0 61
#define BINARY_ELEMENTWISE_CASES_4_0 125
#define BINARY_ELEMENTWISE_CASES BINARY_ELEMENTWISE_CASES_0_0 + \
                                 BINARY_ELEMENTWISE_CASES_1_0 + \
                                 BINARY_ELEMENTWISE_CASES_2_0 + \
                                 BINARY_ELEMENTWISE_CASES_3_0 + \
                                 BINARY_ELEMENTWISE_CASES_4_0

std::vector<int64_t> binary_elementwise_shapes_x[BINARY_ELEMENTWISE_CASES] = {
    // Cases 0.0
    {},
    {},
    {1},
    {1},
    {1},
    {2},
    {2},
    {},
    {2},
    // Cases 1.0
    {},
    {1},
    {1, 1},
    {2},
    {1, 2},
    {3, 1},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 1},
    {1, 2},
    // Cases 2.0
    {3, 2, 1},
    {1, 2, 3},
    {},
    {1},
    {3},
    {2, 3},
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3},
    {1, 1, 1},
    {1, 2, 1},
    {1, 1, 3},
    {4, 1, 1},
    {4, 1, 3},
    {1, 2, 3},
    {4, 2, 1},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    // Cases 3.0
    {},
    {1},
    {2},
    {1, 1},
    {1, 2},
    {3, 1},
    {3, 2},
    {1, 1, 1},
    {4, 1, 1},
    {1, 3, 1},
    {1, 1, 2},
    {4, 3, 1},
    {4, 1, 2},
    {1, 3, 2},
    {4, 3, 2},
    {1, 1, 1, 1},
    {5, 1, 1, 1},
    {1, 4, 1, 1},
    {1, 1, 3, 1},
    {1, 1, 1, 2},
    {5, 4, 1, 1},
    {5, 1, 3, 1},
    {5, 1, 1, 2},
    {1, 4, 3, 1},
    {1, 4, 1, 2},
    {1, 1, 3, 2},
    {5, 4, 3, 1},
    {5, 4, 1, 2},
    {5, 1, 3, 2},
    {1, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    // Cases 4.0
    {},
    {1},
    {2},
    {1, 1},
    {1, 2},
    {3, 1},
    {3, 2},
    {1, 1, 1},
    {4, 1, 1},
    {1, 3, 1},
    {1, 1, 2},
    {4, 3, 1},
    {4, 1, 2},
    {1, 3, 2},
    {4, 3, 2},
    {1, 1, 1, 1},
    {5, 1, 1, 1},
    {1, 4, 1, 1},
    {1, 1, 3, 1},
    {1, 1, 1, 2},
    {5, 4, 1, 1},
    {5, 1, 3, 1},
    {5, 1, 1, 2},
    {1, 4, 3, 1},
    {1, 4, 1, 2},
    {1, 1, 3, 2},
    {5, 4, 3, 1},
    {5, 4, 1, 2},
    {5, 1, 3, 2},
    {1, 4, 3, 2},
    {5, 4, 3, 2},
    {1, 1, 1, 1, 1},
    {6, 1, 1, 1, 1},
    {1, 5, 1, 1, 1},
    {1, 1, 4, 1, 1},
    {1, 1, 1, 3, 1},
    {1, 1, 1, 1, 2},
    {6, 5, 1, 1, 1},
    {6, 1, 4, 1, 1},
    {6, 1, 1, 3, 1},
    {6, 1, 1, 1, 2},
    {1, 5, 4, 1, 1},
    {1, 5, 1, 3, 1},
    {1, 5, 1, 1, 2},
    {1, 1, 4, 3, 1},
    {1, 1, 4, 1, 2},
    {1, 1, 1, 3, 2},
    {6, 5, 4, 1, 1},
    {6, 5, 1, 3, 1},
    {6, 5, 1, 1, 2},
    {6, 1, 4, 3, 1},
    {6, 1, 4, 1, 2},
    {6, 1, 1, 3, 2},
    {1, 5, 4, 3, 1},
    {1, 5, 4, 1, 2},
    {1, 5, 1, 3, 2},
    {1, 1, 4, 3, 2},
    {6, 5, 4, 3, 1},
    {6, 5, 4, 1, 2},
    {6, 5, 1, 3, 2},
    {6, 1, 4, 3, 2},
    {1, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
};

std::vector<int64_t> binary_elementwise_shapes_y[BINARY_ELEMENTWISE_CASES] = {
    // Cases 0.0
    {},
    {1},
    {},
    {1},
    {2},
    {1},
    {},
    {2},
    {2},
    // Cases 1.0
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {3, 2},
    {},
    {1},
    {1, 1},
    {2},
    {1, 2},
    {3, 1},
    {3, 2},
    {1, 2},
    {3, 1},
    // Cases 2.0
    {1, 2, 3},
    {3, 2, 1},
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3},
    {},
    {1},
    {3},
    {2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {4, 2, 3},
    {1, 1, 1},
    {1, 2, 1},
    {1, 1, 3},
    {4, 1, 1},
    {4, 1, 3},
    {1, 2, 3},
    {4, 2, 1},
    {4, 2, 3},
    // Cases 3.0
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {5, 4, 3, 2},
    {},
    {1},
    {2},
    {1, 1},
    {1, 2},
    {3, 1},
    {3, 2},
    {1, 1, 1},
    {4, 1, 1},
    {1, 3, 1},
    {1, 1, 2},
    {4, 3, 1},
    {4, 1, 2},
    {1, 3, 2},
    {4, 3, 2},
    {1, 1, 1, 1},
    {5, 1, 1, 1},
    {1, 4, 1, 1},
    {1, 1, 3, 1},
    {1, 1, 1, 2},
    {5, 4, 1, 1},
    {5, 1, 3, 1},
    {5, 1, 1, 2},
    {1, 4, 3, 1},
    {1, 4, 1, 2},
    {1, 1, 3, 2},
    {5, 4, 3, 1},
    {5, 4, 1, 2},
    {5, 1, 3, 2},
    {1, 4, 3, 2},
    {5, 4, 3, 2},
    // Cases 4.0
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
    {},
    {1},
    {2},
    {1, 1},
    {1, 2},
    {3, 1},
    {3, 2},
    {1, 1, 1},
    {4, 1, 1},
    {1, 3, 1},
    {1, 1, 2},
    {4, 3, 1},
    {4, 1, 2},
    {1, 3, 2},
    {4, 3, 2},
    {1, 1, 1, 1},
    {5, 1, 1, 1},
    {1, 4, 1, 1},
    {1, 1, 3, 1},
    {1, 1, 1, 2},
    {5, 4, 1, 1},
    {5, 1, 3, 1},
    {5, 1, 1, 2},
    {1, 4, 3, 1},
    {1, 4, 1, 2},
    {1, 1, 3, 2},
    {5, 4, 3, 1},
    {5, 4, 1, 2},
    {5, 1, 3, 2},
    {1, 4, 3, 2},
    {5, 4, 3, 2},
    {1, 1, 1, 1, 1},
    {6, 1, 1, 1, 1},
    {1, 5, 1, 1, 1},
    {1, 1, 4, 1, 1},
    {1, 1, 1, 3, 1},
    {1, 1, 1, 1, 2},
    {6, 5, 1, 1, 1},
    {6, 1, 4, 1, 1},
    {6, 1, 1, 3, 1},
    {6, 1, 1, 1, 2},
    {1, 5, 4, 1, 1},
    {1, 5, 1, 3, 1},
    {1, 5, 1, 1, 2},
    {1, 1, 4, 3, 1},
    {1, 1, 4, 1, 2},
    {1, 1, 1, 3, 2},
    {6, 5, 4, 1, 1},
    {6, 5, 1, 3, 1},
    {6, 5, 1, 1, 2},
    {6, 1, 4, 3, 1},
    {6, 1, 4, 1, 2},
    {6, 1, 1, 3, 2},
    {1, 5, 4, 3, 1},
    {1, 5, 4, 1, 2},
    {1, 5, 1, 3, 2},
    {1, 1, 4, 3, 2},
    {6, 5, 4, 3, 1},
    {6, 5, 4, 1, 2},
    {6, 5, 1, 3, 2},
    {6, 1, 4, 3, 2},
    {1, 5, 4, 3, 2},
    {6, 5, 4, 3, 2},
};

#define MATRIX_MULTIPLICATION_CASES_0_0 4
#define MATRIX_MULTIPLICATION_CASES_1_0 10
#define MATRIX_MULTIPLICATION_CASES_2_0 8
#define MATRIX_MULTIPLICATION_CASES_3_0 0
#define MATRIX_MULTIPLICATION_CASES MATRIX_MULTIPLICATION_CASES_0_0 + \
                                    MATRIX_MULTIPLICATION_CASES_1_0 + \
                                    MATRIX_MULTIPLICATION_CASES_2_0 + \
                                    MATRIX_MULTIPLICATION_CASES_3_0


std::vector<int64_t> matrix_multiplication_shapes_x[MATRIX_MULTIPLICATION_CASES] = {
    // Cases 0.0
    {1, 1},
    {2, 1},
    {1, 2},
    {2, 2},
    // Cases 1.0
    {1, 2, 2},
    {2, 2},
    {2, 2, 2},
    {2, 2, 1},
    {1, 2, 1},
    {2, 2, 2},
    {2, 2, 2},
    {2, 2},
    {1, 2, 3},
    {2, 3, 1},
    // Cases 2.0
    {5, 4, 3, 2},
    {5, 4, 2, 3},
    {3, 2},
    {5, 4, 2, 3},
    {1, 1, 2, 3},
    {5, 4, 2, 3},
    {5, 1, 2, 3},
    {1, 4, 2, 3},
};

std::vector<int64_t> matrix_multiplication_shapes_y[MATRIX_MULTIPLICATION_CASES] = {
    // Cases 0.0
    {1, 1},
    {1, 2},
    {2, 1},
    {2, 2},
    // Cases 1.0
    {2, 2},
    {1, 2, 2},
    {2, 2},
    {1, 1, 2},
    {2, 1, 2},
    {2, 2, 2},
    {2, 2},
    {2, 2, 2},
    {2, 3, 1},
    {2, 1, 3},
    // Cases 2.0
    {5, 4, 2, 3},
    {5, 4, 3, 2},
    {5, 4, 2, 3},
    {3, 2},
    {5, 4, 3, 2},
    {1, 1, 3, 2},
    {1, 4, 3, 2},
    {5, 1, 3, 2},
};

nw_error_t *error = NULL;

std::vector<tensor_t *> tensors_x[RUNTIMES][DATATYPES];
std::vector<tensor_t *> tensors_y[RUNTIMES][DATATYPES];
std::vector<tensor_t *> returned_tensors[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_tensors[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_x[RUNTIMES][DATATYPES];
std::vector<tensor_t *> expected_gradients_y[RUNTIMES][DATATYPES];

std::vector<torch::Tensor> torch_tensors_x[RUNTIMES][DATATYPES];
std::vector<torch::Tensor> torch_tensors_y[RUNTIMES][DATATYPES];

typedef enum binary_operation_class_t
{
    BINARY_ELEMENTWISE_CLASS,
    MATRIX_MULTIPLICATION_CLASS,
} binary_operation_class_t;

typedef enum tensor_binary_operation_type_t
{
    TENSOR_ADDITION,
    TENSOR_SUBTRACTION,
    TENSOR_MULTIPLICATION,
    TENSOR_DIVISION,
    TENSOR_POWER,
    TENSOR_MATRIX_MULTIPLICATION,
    TENSOR_COMPARE_EQUAL,
    TENSOR_COMPARE_GREATER,
} tensor_reduction_operation_type_t;

int cases(binary_operation_class_t binary_operation_class)
{
    switch(binary_operation_class)
    {
    case MATRIX_MULTIPLICATION_CLASS:
        return MATRIX_MULTIPLICATION_CASES;
    case BINARY_ELEMENTWISE_CLASS:
        return BINARY_ELEMENTWISE_CASES;
    default:
        return 0;
    }
}

std::vector<int64_t> shapes_x(binary_operation_class_t binary_operation_class, int i)
{
    switch(binary_operation_class)
    {
    case MATRIX_MULTIPLICATION_CLASS:
        return matrix_multiplication_shapes_x[i];
    case BINARY_ELEMENTWISE_CLASS:
        return binary_elementwise_shapes_x[i];
    default:
        return std::vector<int64_t>{};
    }
}

std::vector<int64_t> shapes_y(binary_operation_class_t binary_operation_class, int i)
{
    switch(binary_operation_class)
    {
    case MATRIX_MULTIPLICATION_CLASS:
        return matrix_multiplication_shapes_y[i];
    case BINARY_ELEMENTWISE_CLASS:
        return binary_elementwise_shapes_y[i];
    default:
        return std::vector<int64_t>{};
    }
}

void setup(binary_operation_class_t binary_operation_class)
{
    const int CASES = cases(binary_operation_class);
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            tensors_x[i][j] = std::vector<tensor_t *>(CASES);
            tensors_y[i][j] = std::vector<tensor_t *>(CASES);
            returned_tensors[i][j] = std::vector<tensor_t *>(CASES);
            expected_tensors[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_x[i][j] = std::vector<tensor_t *>(CASES);
            expected_gradients_y[i][j] = std::vector<tensor_t *>(CASES);
            torch_tensors_x[i][j] = std::vector<torch::Tensor>(CASES);
            torch_tensors_y[i][j] = std::vector<torch::Tensor>(CASES);

            for (int k = 0; k < CASES; ++k)
            {
                tensors_x[i][j][k] = NULL;
                tensors_y[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradients_x[i][j][k] = NULL;
                expected_gradients_y[i][j][k] = NULL;

                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x(binary_operation_class, k),
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y(binary_operation_class, k),
                                                            torch::TensorOptions().
                                                            dtype(torch::kFloat32).
                                                            requires_grad(true));
                    break;
                case FLOAT64:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x(binary_operation_class, k),
                                                            torch::TensorOptions().
                                                            dtype(torch::kFloat64).
                                                            requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y(binary_operation_class, k),
                                                            torch::TensorOptions().
                                                            dtype(torch::kFloat64).
                                                            requires_grad(true));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors_x[i][j][k].retain_grad();
                torch_tensors_y[i][j][k].retain_grad();

                tensors_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k], (runtime_t) i, (datatype_t) j);
                tensors_y[i][j][k] = torch_to_tensor(torch_tensors_y[i][j][k], (runtime_t) i, (datatype_t) j);
            }
        }
    }
}

void setup_binary_elementwise(void)
{
    setup(BINARY_ELEMENTWISE_CLASS);
}

void setup_matrix_multiplication(void)
{
    setup(MATRIX_MULTIPLICATION_CLASS);
}

void teardown(binary_operation_class_t binary_operation_class)
{
    const int CASES = cases(binary_operation_class);
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                tensor_destroy(tensors_x[i][j][k]);
                tensor_destroy(tensors_y[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
                tensor_destroy(expected_gradients_x[i][j][k]);
                tensor_destroy(expected_gradients_y[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void teardown_binary_elementwise(void)
{
    teardown(BINARY_ELEMENTWISE_CLASS);
}

void teardown_matrix_multiplication(void)
{
    teardown(MATRIX_MULTIPLICATION_CLASS);
}

void test_binary(binary_operation_class_t binary_operation_class,
                 tensor_binary_operation_type_t tensor_binary_operation_type,
                 bool_t test_gradient)
{
    const int CASES = cases(binary_operation_class);
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;

                switch (tensor_binary_operation_type)
                {
                case TENSOR_ADDITION:
                    expected_tensor = torch::add(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case TENSOR_SUBTRACTION:
                    expected_tensor = torch::sub(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case TENSOR_MULTIPLICATION:
                    expected_tensor = torch::mul(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case TENSOR_DIVISION:
                    expected_tensor = torch::div(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case TENSOR_POWER:
                    expected_tensor = torch::pow(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case TENSOR_MATRIX_MULTIPLICATION:
                    expected_tensor = torch::matmul(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                case TENSOR_COMPARE_EQUAL:
                    expected_tensor = torch::isclose(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k], 1e-6, 1e-9);
                    break;
                case TENSOR_COMPARE_GREATER:
                    expected_tensor = torch::gt(torch_tensors_x[i][j][k], torch_tensors_y[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }

                expected_tensors[i][j][k] = torch_to_tensor(expected_tensor, (runtime_t) i, (datatype_t) j);

                switch (tensor_binary_operation_type)
                {
                case TENSOR_ADDITION:
                    error = tensor_addition(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_SUBTRACTION:
                    error = tensor_subtraction(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_MULTIPLICATION:
                    error = tensor_multiplication(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_DIVISION:
                    error = tensor_division(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_POWER:
                    error = tensor_power(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_MATRIX_MULTIPLICATION:
                    error = tensor_matrix_multiplication(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_COMPARE_EQUAL:
                    error = tensor_compare_equal(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                case TENSOR_COMPARE_GREATER:
                    error = tensor_compare_greater(tensors_x[i][j][k], tensors_y[i][j][k], &returned_tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(returned_tensors[i][j][k], expected_tensors[i][j][k]);

                if (!test_gradient)
                {
                    tensor_destroy(returned_tensors[i][j][k]);
                    continue;
                }

                expected_tensor.sum().backward();
                expected_gradients_x[i][j][k] = torch_to_tensor(torch_tensors_x[i][j][k].grad(), (runtime_t) i, (datatype_t) j);
                expected_gradients_y[i][j][k] = torch_to_tensor(torch_tensors_y[i][j][k].grad(), (runtime_t) i, (datatype_t) j);

                tensor_t *cost;
                error = tensor_summation(returned_tensors[i][j][k], &cost, NULL, 0, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors_x[i][j][k]->gradient, expected_gradients_x[i][j][k]);
                ck_assert_tensor_equiv(tensors_y[i][j][k]->gradient, expected_gradients_y[i][j][k]);
            }
        }
    }
}

START_TEST(test_addition)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_ADDITION, true);
}
END_TEST

START_TEST(test_subtraction)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_SUBTRACTION, true);
}
END_TEST

START_TEST(test_multiplication)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_MULTIPLICATION, true);
}
END_TEST

START_TEST(test_division)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_DIVISION, true);
}
END_TEST

START_TEST(test_power)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_POWER, true);
}
END_TEST

START_TEST(test_compare_equal)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_COMPARE_EQUAL, false);
}
END_TEST

START_TEST(test_compare_greater)
{
    test_binary(BINARY_ELEMENTWISE_CLASS, TENSOR_COMPARE_GREATER, false);
}
END_TEST

START_TEST(test_matrix_multiplication)
{
    test_binary(MATRIX_MULTIPLICATION_CLASS, TENSOR_MATRIX_MULTIPLICATION, true);
}
END_TEST

Suite *make_binary_suite(void)
{
    Suite *s;
    TCase *tc_binary_elementwise;
    TCase *tc_matrix_multiplication;

    s = suite_create("Test Binary Tensor Suite");

    tc_binary_elementwise = tcase_create("Test Binary Elementwise Case");
    tcase_add_checked_fixture(tc_binary_elementwise, setup_binary_elementwise, teardown_binary_elementwise);
    tcase_add_test(tc_binary_elementwise, test_addition);
    tcase_add_test(tc_binary_elementwise, test_subtraction);
    tcase_add_test(tc_binary_elementwise, test_multiplication);
    tcase_add_test(tc_binary_elementwise, test_division);
    tcase_add_test(tc_binary_elementwise, test_power);
    tcase_add_test(tc_binary_elementwise, test_compare_equal);
    tcase_add_test(tc_binary_elementwise, test_compare_greater);

    tc_matrix_multiplication = tcase_create("Test Matrix Multiplication Case");
    tcase_add_checked_fixture(tc_matrix_multiplication, setup_matrix_multiplication, teardown_matrix_multiplication);
    tcase_add_test(tc_matrix_multiplication, test_matrix_multiplication);

    suite_add_tcase(s, tc_binary_elementwise);
    suite_add_tcase(s, tc_matrix_multiplication);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_binary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}