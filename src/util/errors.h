/**@file errors.h
 * @brief Provides error types and utilities that interact with them.
 *
 */

#ifndef ERRORS_H
#define ERRORS_H

#include <stdio.h>
#include <datatype.h>

#ifdef DEBUG
#define PRINT_DEBUG_NEWLINE do {\
    fprintf(stderr, "\n");\
} while(0)

#define PRINT_DEBUG_LOCATION do {\
    fprintf(stderr, "(");\
    fprintf(stderr, "function: %s", __FUNCTION__);\
    fprintf(stderr, ", line: %d", __LINE__);\
    fprintf(stderr, ", file: %s", __FILE__);\
    fprintf(stderr, ")");\
} while(0)

#define PRINTLN_DEBUG_LOCATION(msg) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_LOCATION;\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_BOOLEAN(boolean) do {\
    fprintf(stderr, "%s", boolean ? "true" : "false");\
} while(0)

#define PRINTLN_DEBUG_BOOLEAN(msg, boolean) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_BOOLEAN(boolean);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINTF_DEBUG(format, ...) do {\
    fprintf(stderr, format, __VA_ARGS__);\
} while(0)

#define PRINT_DEBUG(string) do {\
    fprintf(stderr, string);\
} while(0)

#define PRINT_DEBUG_UINT64_ARRAY(array, length) do {\
    if (array == NULL)\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        for (uint64_t i = 0; i < (length); ++i)\
        {\
            if (!i)\
            {\
                fprintf(stderr, "%lu", array[i]);\
            }\
            else\
            {\
                fprintf(stderr, ", %lu", array[i]);\
            }\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_UINT64_ARRAY(msg, array, length) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_UINT64_ARRAY(array, length);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_VIEW(view) do {\
    if (view == NULL)\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(offset: %lu", view->offset);\
        fprintf(stderr, ", rank: %lu", view->rank);\
        fprintf(stderr, ", shape: ");\
        PRINT_DEBUG_UINT64_ARRAY(view->shape, view->rank);\
        fprintf(stderr, ", strides: ");\
        PRINT_DEBUG_UINT64_ARRAY(view->strides, view->rank);\
        fprintf(stderr, ")");\
    }\
} while(0)
 
#define PRINTLN_DEBUG_VIEW(msg, view) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_VIEW(view);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_BUFFER(buffer) do {\
    if (buffer == NULL)\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(runtime: %s", runtime_string(buffer->runtime));\
        fprintf(stderr, ", datatype: %s", datatype_string(buffer->datatype));\
        fprintf(stderr, ", view: ");\
        PRINT_DEBUG_VIEW(buffer->view);\
        fprintf(stderr, ", size: %zu", buffer->size);\
        fprintf(stderr, ", n: %lu", buffer->n);\
        fprintf(stderr, ", copy: ");\
        PRINT_DEBUG_BOOLEAN(buffer->copy);\
        fprintf(stderr, ", data: ");\
        if (buffer->data == NULL)\
        {\
            fprintf(stderr, "NULL");\
        }\
        else\
        {\
            fprintf(stderr, "(");\
            for (uint64_t i = 0; i < buffer->n; ++i)\
            {\
                switch(buffer->datatype)\
                {\
                case FLOAT32:\
                    if (!i)\
                    {\
                        fprintf(stderr, "%f", ((float *) buffer->data)[i]);\
                    }\
                    else\
                    {\
                        fprintf(stderr, ", %f", ((float *) buffer->data)[i]);\
                    }\
                    break;\
                case FLOAT64:\
                    if (!i)\
                    {\
                        fprintf(stderr, "%lf", ((double *) buffer->data)[i]);\
                    }\
                    else\
                    {\
                        fprintf(stderr, ", %lf", ((double *) buffer->data)[i]);\
                    }\
                    break;\
                default:\
                    break;\
                }\
            }\
            fprintf(stderr, ")");\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_BUFFER(msg, buffer) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_BUFFER(buffer);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_FUNCTION(function) do {\
    if (function == NULL)\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(operation_type: %s", operation_type_string(function->operation_type));\
        fprintf(stderr, ", operation: ");\
        switch(function->operation_type)\
        {\
        case UNARY_OPERATION:\
            fprintf(stderr, "%s", unary_operation_type_string(function->operation->unary_operation->operation_type));\
            if (function->operation->unary_operation->x == NULL)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", function->operation->unary_operation->x->id);\
            }\
            if (function->operation->unary_operation->result == NULL)\
            {\
                fprintf(stderr, ", result: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", result: (id: %lu)", function->operation->unary_operation->result->id);\
            }\
            break;\
        case BINARY_OPERATION:\
            fprintf(stderr, "%s", binary_operation_type_string(function->operation->binary_operation->operation_type));\
            if (function->operation->binary_operation->x == NULL)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", function->operation->binary_operation->x->id);\
            }\
            if (function->operation->binary_operation->y == NULL)\
            {\
                fprintf(stderr, ", y: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", y: (id: %lu)", function->operation->binary_operation->y->id);\
            }\
            if (function->operation->binary_operation->result == NULL)\
            {\
                fprintf(stderr, ", result: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", result: (id: %lu)", function->operation->binary_operation->result->id);\
            }\
            break;\
        case REDUCTION_OPERATION:\
            fprintf(stderr, "%s", reduction_operation_type_string(function->operation->reduction_operation->operation_type));\
            if (function->operation->reduction_operation->x == NULL)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", function->operation->reduction_operation->x->id);\
            }\
            if (function->operation->reduction_operation->result == NULL)\
            {\
                fprintf(stderr, ", result: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", result: (id: %lu)", function->operation->reduction_operation->result->id);\
            }\
            fprintf(stderr, ", axis: ");\
            PRINT_DEBUG_UINT64_ARRAY(function->operation->reduction_operation->axis,\
                                     function->operation->reduction_operation->length);\
            fprintf(stderr, ", keep_dimension: ");\
            PRINT_DEBUG_BOOLEAN(function->operation->reduction_operation->keep_dimension);\
            break;\
        case STRUCTURE_OPERATION:\
            fprintf(stderr, "%s", structure_operation_type_string(function->operation->structure_operation->operation_type));\
            if (function->operation->structure_operation->x == NULL)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", function->operation->structure_operation->x->id);\
            }\
            if (function->operation->structure_operation->result == NULL)\
            {\
                fprintf(stderr, ", result: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", result: (id: %lu)", function->operation->structure_operation->result->id);\
            }\
            fprintf(stderr, ", arguments: ");\
            PRINT_DEBUG_UINT64_ARRAY(function->operation->structure_operation->arguments,\
                                     function->operation->structure_operation->length);\
        default:\
            break;\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_FUNCTION(msg, function) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_FUNCTION(function);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_TENSOR(tensor) do {\
    if (tensor == NULL)\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(id: %lu", tensor->id);\
        fprintf(stderr, ", buffer: ");\
        PRINT_DEBUG_BUFFER(tensor->buffer);\
        fprintf(stderr, ", context: ");\
        PRINT_DEBUG_FUNCTION(tensor->context);\
        if (tensor->gradient == NULL)\
        {\
            fprintf(stderr, ", gradient: NULL");\
        }\
        else\
        {\
            fprintf(stderr, ", gradient: (id: %lu)", tensor->gradient->id);\
        }\
        fprintf(stderr, ", requires_gradient: ");\
        PRINT_DEBUG_BOOLEAN(tensor->requires_gradient);\
        fprintf(stderr, ", lock: ");\
        PRINT_DEBUG_BOOLEAN(tensor->lock);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_TENSOR(msg, tensor) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_TENSOR(tensor);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#else
#define PRINT_DEBUG(format, ...)
#define PRINTF_DEBUG(format, ...)
#define PRINT_DEBUG_UINT64_ARRAY(array, length)
#define PRINTLN_DEBUG_UINT64_ARRAY(msg, array, length)
#define PRINT_DEBUG_VIEW(view)
#define PRINTLN_DEBUG_VIEW(msg, view)
#define PRINT_DEBUG_BUFFER(buffer)
#define PRINTLN_DEBUG_BUFFER(msg, buffer)
#define PRINT_DEBUG_FUNCTION(function)
#define PRINTLN_DEBUG_FUNCTION(msg, function)
#define PRINT_DEBUG_TENSOR(tensor)
#define PRINTLN_DEBUG_TENSOR(msg, tensor)
#define PRINT_DEBUG_LOCATION
#define PRINTLN_DEBUG_LOCATION(msg)
#define PRINT_DEBUG_NEWLINE
#define PRINT_DEBUG_BOOLEAN(boolean)
#define PRINTLN_DEBUG_BOOLEAN(msg, boolean)
#endif

typedef enum nw_error_type_t
{
    ERROR_MEMORY_ALLOCATION,
    ERROR_MEMORY_FREE,
    ERROR_UNKNOWN_RUNTIME,
    ERROR_UKNOWN_OPERATION_TYPE,
    ERROR_NULL,
    ERROR_DATATYPE_CONFLICT,
    ERROR_SHAPE_CONFLICT,
    ERROR_RUNTIME_CONFLICT,
    ERROR_RANK_CONFLICT,
    ERROR_CREATE,
    ERROR_DESTROY,
    ERROR_BROADCAST,
    ERROR_INITIALIZATION,
    ERROR_DATATYPE,
    ERROR_COPY,
    ERROR_ADDITION,
    ERROR_CONTIGUOUS,
    ERROR_FORWARD,
    ERROR_BACKWARD,
    ERROR_SET,
    ERROR_OVERFLOW,
    ERROR_EXPAND,
    ERROR_PERMUTE,
    ERROR_SUMMATION,
    ERROR_SQUARE_ROOT,
    ERROR_RESHAPE,
    ERROR_SLICE,
    ERROR_BINARY_ELEMENTWISE,
    ERROR_REDUCTION,
    ERROR_EXPONENTIAL,
    ERROR_LOGARITHM,
    ERROR_DIVISION,
    ERROR_SINE,
    ERROR_COSINE,
    ERROR_RECIPROCAL,
    ERROR_MULTIPLICATION,
    ERROR_NEGATION,
    ERROR_SUBTRACTION,
    ERROR_POWER,
    ERROR_MATRIX_MULTIPLICATION,
    ERROR_COMPARE_EQUAL,
    ERROR_COMPARE_GREATER,
    ERROR_PADDING,
    ERROR_RECTIFIED_LINEAR,
    ERROR_AXIS,
    ERROR_SORT,
    ERROR_POP,
    ERROR_MAXIMUM,
} nw_error_type_t;

typedef struct nw_error_t
{
    nw_error_type_t error_type;
    string_t file;
    uint64_t line_number;
    string_t function;
    string_t message;
    struct nw_error_t *next_error;
} nw_error_t;

nw_error_t *error_create(nw_error_type_t error_type, string_t file, uint64_t line_number, string_t function, string_t message, nw_error_t *next_error);
void error_destroy(nw_error_t *error);
void error_print(nw_error_t *error);
string_t error_type_string(nw_error_type_t error_type);

#define ERROR(error_type, error_string, error) (error_create(error_type, __FILE__, __LINE__, __FUNCTION__, error_string, error))

#define CHECK_NULL_ARGUMENT(pointer, string) do {\
            if (pointer == NULL)\
            {\
                return ERROR(ERROR_NULL, string_create("received null argument for %s.", string), NULL);\
            }\
        } while (0)

#endif
