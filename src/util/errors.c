/**@file errors.c
 * @brief Implements utilities that interact with error types.
 *
 */

#include <errors.h>

nw_error_t *error_create(nw_error_type_t error_type, string_t file, uint64_t line_number, string_t function, string_t message, nw_error_t *next_error)
{
    nw_error_t *error = (nw_error_t *) malloc(sizeof(nw_error_t));
    if (!error)
        return NULL;

    error->error_type = error_type;
    error->file = file;
    error->line_number = line_number;
    error->function = function;
    error->message = message;
    error->next_error = next_error;
    return error;
}

void error_destroy(nw_error_t *error)
{
    while(error)
    {
        nw_error_t *next_error = error->next_error;
        string_destroy(error->message);
        free(error);
        error = next_error;
    }
}

string_t error_type_string(nw_error_type_t error_type)
{
    switch (error_type)
    {
    case ERROR_MEMORY_ALLOCATION:
        return "ERROR_MEMORY_ALLOCATION"; 
    case ERROR_MEMORY_FREE:
        return "ERROR_MEMORY_FREE";
    case ERROR_UNKNOWN_RUNTIME:
        return "ERROR_UNKNOWN_RUNTIME";
    case ERROR_UKNOWN_OPERATION_TYPE:
        return "ERROR_UKNOWN_OPERATION_TYPE";
    case ERROR_UKNOWN_LAYER_TYPE:
        return "ERROR_UKNOWN_LAYER_TYPE";
    case ERROR_NULL:
        return "ERROR_NULL";
    case ERROR_DATATYPE_CONFLICT:
        return "ERROR_DATATYPE_CONFLICT";
    case ERROR_SHAPE_CONFLICT:
        return "ERROR_SHAPE_CONFLICT";
    case ERROR_RUNTIME_CONFLICT:
        return "ERROR_RUNTIME_CONFLICT";
    case ERROR_RANK_CONFLICT:
        return "ERROR_RANK_CONFLICT";
    case ERROR_CREATE:
        return "ERROR_CREATE";
    case ERROR_DESTROY:
        return "ERROR_DESTROY";
    case ERROR_BROADCAST:
        return "ERROR_BROADCAST";
    case ERROR_INITIALIZATION:
        return "ERROR_INITIALIZATION";
    case ERROR_DATATYPE:
        return "ERROR_DATATYPE";
    case ERROR_COPY:
        return "ERROR_COPY";
    case ERROR_ADDITION:
        return "ERROR_ADDITION";
    case ERROR_CONTIGUOUS:
        return "ERROR_CONTIGUOUS";
    case ERROR_FORWARD:
        return "ERROR_FORWARD";
    case ERROR_BACKWARD:
        return "ERROR_BACKWARD";
    case ERROR_SET:
        return "ERROR_SET";
    case ERROR_OVERFLOW:
        return "ERROR_OVERFLOW";
    case ERROR_EXPAND:
        return "ERROR_EXPAND";
    case ERROR_PERMUTE:
        return "ERROR_PERMUTE";
    case ERROR_SUMMATION:
        return "ERROR_SUMMATION";
    case ERROR_SQUARE_ROOT:
        return "ERROR_SQUARE_ROOT";
    case ERROR_RESHAPE:
        return "ERROR_RESHAPE";
    case ERROR_SLICE:
        return "ERROR_SLICE";
    case ERROR_BINARY_ELEMENTWISE:
        return "ERROR_BINARY_ELEMENTWISE";
    case ERROR_REDUCTION:
        return "ERROR_REDUCTION";
    case ERROR_EXPONENTIAL:
        return "ERROR_EXPONENTIAL";
    case ERROR_LOGARITHM:
        return "ERROR_LOGARITHM";
    case ERROR_DIVISION:
        return "ERROR_DIVISION";
    case ERROR_SINE:
        return "ERROR_SINE";
    case ERROR_COSINE:
        return "ERROR_COSINE";
    case ERROR_RECIPROCAL:
        return "ERROR_RECIPROCAL";
    case ERROR_NEGATION:
        return "ERROR_NEGATION";
    case ERROR_MATRIX_MULTIPLICATION:
        return "ERROR_MATRIX_MULTIPLICATION";
    case ERROR_COMPARE_EQUAL:
        return "ERROR_COMPARE_EQUAL";
    case ERROR_COMPARE_GREATER:
        return "ERROR_COMPARE_GREATER";
    case ERROR_MAX:
        return "ERROR_MAX";
    case ERROR_PADDING:
        return "ERROR_PADDING";
    case ERROR_RECTIFIED_LINEAR:
        return "ERROR_RECTIFIED_LINEAR";
    case ERROR_AXIS:
        return "ERROR_AXIS";
    case ERROR_SORT:
        return "ERROR_SORT";
    case ERROR_POP:
        return "ERROR_POP";
    case ERROR_MAXIMUM:
        return "ERROR_MAXIMUM";
    case ERROR_UNIQUE:
        return "ERROR_UNIQUE";
    case ERROR_UNARY:
        return "ERROR_UNARY";
    case ERROR_SIGMOID:
        return "ERROR_SIGMOID";
    case ERROR_PUSH:
        return "ERROR_PUSH";
    case ERROR_TRANSPOSE:
        return "ERROR_TRANSPOSE";
    case ERROR_SOFTMAX:
        return "ERROR_SOFTMAX";
    case ERROR_MEAN:
        return "ERROR_MEAN";
    case ERROR_FILE:
        return "ERROR_FILE";
    case ERROR_SETUP:
        return "ERROR_SETUP";
    case ERROR_TEARDOWN:
        return "ERROR_TEARDOWN";
    case ERROR_LOAD:
        return "ERROR_LOAD";
    case ERROR_CRITERION:
        return "ERROR_CRITERION";
    case ERROR_METRICS:
        return "ERROR_METRICS";
    case ERROR_STEP:
        return "ERROR_STEP";
    case ERROR_RESET:
        return "ERROR_RESET";
    case ERROR_TRAIN:
        return "ERROR_TRAIN";
    case ERROR_VALID:
        return "ERROR_VALID";
    case ERROR_TEST:
        return "ERROR_TEST";
    case ERROR_UNKNOWN_ALGORITHM:
        return "ERROR_UNKNOWN_ALGORITHM";
    case ERROR_GAIN:
        return "ERROR_GAIN";
    case ERROR_UPDATE:
        return "ERROR_UPDATE";
    case ERROR_REQUIRES_GRADIENT:
        return "ERROR_REQUIRES_GRADIENT";
    case ERROR_ACTIVATION_TYPE:
        return "ERROR_ACTIVATION_TYPE";
    case ERROR_DROPOUT:
        return "ERROR_DROPOUT";
    case ERROR_DATASET_TYPE:
        return "ERROR_DATASET_TYPE";
    case ERROR_ARGUMENTS:
        return "ERROR_ARGUMENTS";
    case ERROR_ITEM:
        return "ERROR_ITEM";
    case ERROR_OPTIM:
        return "ERROR_OPTIM";
    default:
        return "ERROR";
    }
}

void error_print(nw_error_t *error)
{
    while (error)
    {
        fprintf(
            stderr, 
            "%s:%s:%lu:%s:%s\n", 
            error_type_string(error->error_type),
            error->file ? error->file : "NULL", 
            error->line_number, 
            error->function ? error->function : "NULL", 
            error->message ? error->message : "NULL"
        );
        error = error->next_error;
    }
}
