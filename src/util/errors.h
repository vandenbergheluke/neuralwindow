/**@file errors.h
 * @brief Provides error types and utilities that interact with them.
 *
 */

#ifndef ERRORS_H
#define ERRORS_H

#include <stdio.h>
#include <datatype.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ABS(x) ((x)<0 ? -(x) : (x))

#ifdef DEBUG
#define MAX_DATA 1000
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
    fprintf(stderr, "%s", (boolean) ? "true" : "false");\
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

#define PRINT_DEBUG_INT64_ARRAY(array, length) do {\
    if (!(array))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        for (int64_t i = 0; i < (length); ++i)\
        {\
            if (!i)\
            {\
                fprintf(stderr, "%ld", (array)[i]);\
            }\
            else\
            {\
                fprintf(stderr, ", %ld", (array)[i]);\
            }\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_INT64_ARRAY(msg, array, length) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_INT64_ARRAY((array), length);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_VIEW(view) do {\
    if (!(view))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(offset: %ld", (view)->offset);\
        fprintf(stderr, ", rank: %ld", (view)->rank);\
        fprintf(stderr, ", shape: ");\
        PRINT_DEBUG_INT64_ARRAY((view)->shape, (view)->rank);\
        fprintf(stderr, ", strides: ");\
        PRINT_DEBUG_INT64_ARRAY((view)->strides, (view)->rank);\
        fprintf(stderr, ")");\
    }\
} while(0)
 
#define PRINTLN_DEBUG_VIEW(msg, view) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_VIEW((view));\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_STORAGE(storage) do {\
    if (!(storage))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(runtime: %s", runtime_string((storage)->runtime));\
        fprintf(stderr, ", datatype: %s", datatype_string((storage)->datatype));\
        fprintf(stderr, ", n: %ld", (storage)->n);\
        fprintf(stderr, ", reference_count: %ld", (storage)->reference_count);\
        fprintf(stderr, ", data: ");\
        if (!(storage)->data)\
        {\
            fprintf(stderr, "NULL");\
        }\
        else\
        {\
            fprintf(stderr, "(");\
            int64_t n = MIN((storage)->n, MAX_DATA);\
            for (int64_t j = 0; j < n; ++j)\
            {\
                switch((storage)->datatype)\
                {\
                case FLOAT32:\
                    if (!j)\
                    {\
                        fprintf(stderr, "%f", ((float *) (storage)->data)[j]);\
                    }\
                    else\
                    {\
                        fprintf(stderr, ", %f", ((float *) (storage)->data)[j]);\
                    }\
                    break;\
                case FLOAT64:\
                    if (!j)\
                    {\
                        fprintf(stderr, "%lf", ((double *) (storage)->data)[j]);\
                    }\
                    else\
                    {\
                        fprintf(stderr, ", %lf", ((double *) (storage)->data)[j]);\
                    }\
                    break;\
                default:\
                    break;\
                }\
            }\
            if (n != (storage)->n)\
            {\
                fprintf(stderr, ", ...");\
            }\
            fprintf(stderr, ")");\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)
 
#define PRINTLN_DEBUG_STORAGE(msg, storage) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_STORAGE((storage));\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_BUFFER(buffer) do {\
    if (!(buffer))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(view: ");\
        PRINT_DEBUG_VIEW((buffer)->view);\
        fprintf(stderr, ", storage: ");\
        PRINT_DEBUG_STORAGE((buffer)->storage);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_BUFFER(msg, buffer) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_BUFFER((buffer));\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_FUNCTION(function) do {\
    if (!(function))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(operation_type: %s", operation_type_string((function)->operation_type));\
        fprintf(stderr, ", operation: ");\
        switch((function)->operation_type)\
        {\
        case UNARY_OPERATION:\
            fprintf(stderr, "%s", unary_operation_type_string((function)->operation->unary_operation->operation_type));\
            if (!(function)->operation->unary_operation->x)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", (function)->operation->unary_operation->x->id);\
            }\
            break;\
        case BINARY_OPERATION:\
            fprintf(stderr, "%s", binary_operation_type_string((function)->operation->binary_operation->operation_type));\
            if (!(function)->operation->binary_operation->x)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", (function)->operation->binary_operation->x->id);\
            }\
            if (!(function)->operation->binary_operation->y)\
            {\
                fprintf(stderr, ", y: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", y: (id: %lu)", (function)->operation->binary_operation->y->id);\
            }\
            break;\
        case REDUCTION_OPERATION:\
            fprintf(stderr, "%s", reduction_operation_type_string((function)->operation->reduction_operation->operation_type));\
            if (!(function)->operation->reduction_operation->x)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", (function)->operation->reduction_operation->x->id);\
            }\
            fprintf(stderr, ", axis: ");\
            PRINT_DEBUG_INT64_ARRAY((function)->operation->reduction_operation->axis,\
                                     (function)->operation->reduction_operation->length);\
            fprintf(stderr, ", keep_dimension: ");\
            PRINT_DEBUG_BOOLEAN((function)->operation->reduction_operation->keep_dimension);\
            break;\
        case STRUCTURE_OPERATION:\
            fprintf(stderr, "%s", structure_operation_type_string((function)->operation->structure_operation->operation_type));\
            if (!(function)->operation->structure_operation->x)\
            {\
                fprintf(stderr, ", x: NULL");\
            }\
            else\
            {\
                fprintf(stderr, ", x: (id: %lu)", (function)->operation->structure_operation->x->id);\
            }\
            fprintf(stderr, ", arguments: ");\
            PRINT_DEBUG_INT64_ARRAY((function)->operation->structure_operation->arguments,\
                                     (function)->operation->structure_operation->length);\
            break;\
        case CREATION_OPERATION:\
            fprintf(stderr, "%s", creation_operation_type_string((function)->operation->creation_operation->operation_type));\
            fprintf(stderr, ", shape: ");\
            PRINT_DEBUG_INT64_ARRAY((function)->operation->creation_operation->shape,\
                                     (function)->operation->creation_operation->rank);\
            break;\
        default:\
            break;\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_FUNCTION(msg, function) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_FUNCTION((function));\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_TENSOR(tensor) do {\
    if (!(tensor))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(id: %lu", (tensor)->id);\
        fprintf(stderr, ", buffer: ");\
        PRINT_DEBUG_BUFFER((tensor)->buffer);\
        fprintf(stderr, ", context: ");\
        PRINT_DEBUG_FUNCTION((tensor)->context);\
        if (!(tensor)->gradient)\
        {\
            fprintf(stderr, ", gradient: NULL");\
        }\
        else\
        {\
            fprintf(stderr, ", gradient: (id: %lu)", (tensor)->gradient->id);\
        }\
        fprintf(stderr, ", requires_gradient: ");\
        PRINT_DEBUG_BOOLEAN((tensor)->requires_gradient);\
        fprintf(stderr, ", persist: ");\
        PRINT_DEBUG_BOOLEAN((tensor)->persist);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_TENSOR(msg, tensor) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_TENSOR((tensor));\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_SOFTMAX(softmax) do {\
    if (!(softmax))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "axis: %ld", (softmax)->axis);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_SOFTMAX(msg, softmax) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_SOFTMAX(softmax);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_ACTIVATION(activation) do {\
    if (!(activation))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "activation_function_type: %s", activation_function_type_string((activation)->activation_function_type));\
        fprintf(stderr, ", activation_function: ");\
        if (!(activation)->activation_function)\
        {\
            fprintf(stderr, "NULL");\
        }\
        else\
        {\
            fprintf(stderr, "(");\
            switch ((activation)->activation_function_type)\
            {\
            case ACTIVATION_SOFTMAX:\
            case ACTIVATION_LOGSOFTMAX:\
                fprintf(stderr, "softmax: ");\
                PRINT_DEBUG_SOFTMAX((activation)->activation_function->softmax);\
                break;\
            default:\
                break;\
            }\
            fprintf(stderr, ")");\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_ACTIVATION(msg, activation) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_ACTIVATION(activation);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_LINEAR(linear) do {\
    if (!(linear))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "weights: ");\
        PRINT_DEBUG_TENSOR((linear)->weights);\
        fprintf(stderr, ", bias: ");\
        PRINT_DEBUG_TENSOR((linear)->bias);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_LINEAR(msg, linear) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_LINEAR(linear);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_DROPOUT(dropout) do {\
    if (!(dropout))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        switch((dropout)->datatype)\
        {\
        case FLOAT32:\
            fprintf(stderr, "probability: %f", *(float32_t *) ((dropout)->probability));\
            break;\
        case FLOAT64:\
            fprintf(stderr, "probability: %lf", *(float64_t *) ((dropout)->probability));\
            break;\
        default:\
            break;\
        }\
        fprintf(stderr, ", inference: ");\
        PRINT_DEBUG_BOOLEAN((dropout)->inference);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_DROPOUT(msg, dropout) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_DROPOUT(dropout);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_LAYER(layer) do {\
    if (!(layer))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "transform_type: %s", transform_type_string((layer)->transform_type));\
        fprintf(stderr, ", transform: ");\
        if (!(layer)->transform)\
        {\
            fprintf(stderr, "NULL");\
        }\
        else\
        {\
            switch ((layer)->transform_type)\
            {\
            case LINEAR:\
                PRINT_DEBUG_LINEAR((layer)->transform->linear);\
                break;\
            case DROPOUT:\
                PRINT_DEBUG_DROPOUT((layer)->transform->dropout);\
                break;\
            case BLOCK:\
                fprintf(stderr, "(");\
                if (!(layer)->transform->block)\
                {\
                    fprintf(stderr, "NULL");\
                }\
                else\
                {\
                    fprintf(stderr, "depth: %ld", (layer)->transform->block->depth);\
                }\
                fprintf(stderr, ")");\
                break;\
            default:\
                break;\
            }\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_LAYER(msg, layer) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_LAYER(layer);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_BLOCK(block) do {\
    if (!(block))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "depth: %ld", (block)->depth);\
        fprintf(stderr, ", layers: ");\
        if (!(block)->layers)\
        {\
            fprintf(stderr, "NULL");\
        }\
        else\
        {\
            fprintf(stderr, "(");\
            for (int64_t k = 0; k < (block)->depth; ++k)\
            {\
                if (k)\
                {\
                    fprintf(stderr, ", ");\
                }\
                PRINT_DEBUG_LAYER(((block)->layers)[k]);\
            }\
            fprintf(stderr, ")");\
        }\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_BLOCK(msg, block) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_BLOCK(block);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_MODEL(model) do {\
    if (!(model))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "block: ");\
        PRINT_DEBUG_BLOCK((model)->block);\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_MODEL(msg, model) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_MODEL(model);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#define PRINT_DEBUG_OPTIMIZER(optimizer) do {\
    if (!(optimizer))\
    {\
        fprintf(stderr, "NULL");\
    }\
    else\
    {\
        fprintf(stderr, "(");\
        fprintf(stderr, "algorithm_type: %s", algorithm_type_string((optimizer)->algorithm_type));\
        fprintf(stderr, ")");\
    }\
} while(0)

#define PRINTLN_DEBUG_OPTIMIZER(msg, optimizer) do {\
    fprintf(stderr, "%s ", msg);\
    PRINT_DEBUG_OPTIMIZER(optimizer);\
    PRINT_DEBUG_NEWLINE;\
} while(0)

#else
#define PRINT_DEBUG(format, ...)
#define PRINTF_DEBUG(format, ...)
#define PRINT_DEBUG_INT64_ARRAY(array, length)
#define PRINTLN_DEBUG_INT64_ARRAY(msg, array, length)
#define PRINT_DEBUG_VIEW(view)
#define PRINTLN_DEBUG_VIEW(msg, view)
#define PRINT_DEBUG_STORAGE(storage)
#define PRINTLN_DEBUG_STORAGE(msg, storage)
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
#define PRINTLN_DEBUG_SOFTMAX(msg, softmax)
#define PRINT_DEBUG_ACTIVATION(activation)
#define PRINTLN_DEBUG_ACTIVATION(msg, activation)
#define PRINT_DEBUG_LINEAR(linear)
#define PRINTLN_DEBUG_LINEAR(msg, linear)
#define PRINT_DEBUG_DROPOUT(dropout)
#define PRINTLN_DEBUG_DROPOUT(msg, dropout)
#define PRINT_DEBUG_LAYER(layer)
#define PRINTLN_DEBUG_LAYER(msg, layer)
#define PRINT_DEBUG_BLOCK(block)
#define PRINTLN_DEBUG_BLOCK(msg, block)
#define PRINT_DEBUG_MODEL(model)
#define PRINTLN_DEBUG_MODEL(msg, model)
#define PRINT_DEBUG_OPTIMIZER(optimizer)
#define PRINTLN_DEBUG_OPTIMIZER(msg, optimizer)
#endif

typedef enum nw_error_type_t
{
    ERROR_MEMORY_ALLOCATION,
    ERROR_MEMORY_FREE,
    ERROR_RUNTIME,
    ERROR_OPERATION_TYPE,
    ERROR_LAYER_TYPE,
    ERROR_NULL,
    ERROR_DATATYPE,
    ERROR_SHAPE,
    ERROR_RANK,
    ERROR_CREATE,
    ERROR_DESTROY,
    ERROR_BROADCAST,
    ERROR_INITIALIZATION,
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
    ERROR_BINARY,
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
    ERROR_MAX,
    ERROR_RECTIFIED_LINEAR,
    ERROR_AXIS,
    ERROR_SORT,
    ERROR_POP,
    ERROR_MAXIMUM,
    ERROR_UNIQUE,
    ERROR_N,
    ERROR_UNARY,
    ERROR_SIGMOID,
    ERROR_PUSH,
    ERROR_TRANSPOSE,
    ERROR_SOFTMAX,
    ERROR_MEAN,
    ERROR_FILE,
    ERROR_SETUP,
    ERROR_TEARDOWN,
    ERROR_LOAD,
    ERROR_CRITERION,
    ERROR_METRICS,
    ERROR_STEP,
    ERROR_RESET,
    ERROR_TRAIN,
    ERROR_VALID,
    ERROR_TEST,
    ERROR_ALGORITHM,
    ERROR_GAIN,
    ERROR_UPDATE,
    ERROR_REQUIRES_GRADIENT,
    ERROR_ACTIVATION_TYPE,
    ERROR_DROPOUT,
    ERROR_DATASET_TYPE,
    ERROR_GRAPH,
    ERROR_ARGUMENTS,
    ERROR_ITEM,
    ERROR_OPTIM,
    ERROR_POOLING,
    ERROR_CONVOLUTION,
    ERROR_IMAGE_TO_COLUMN,
    ERROR_COLUMN_TO_IMAGE,
    ERROR_PADDING,
    ERROR_SLICE,
    ERROR_GET,
    ERROR_LINEAR,
    ERROR_VARIANCE,
    ERROR_STANDARD_DEVIATION,
    ERROR_BATCH_NORMALIZATION,
    ERROR_ABSOLUTE,
    ERROR_FAN,
    ERROR_ZERO_GRADIENT,
    ERROR_LOWER_TRIANGULAR,
    ERROR_ATTENTION,
    ERROR_WHERE,
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
            if (!pointer)\
            {\
                return ERROR(ERROR_NULL, string_create("received null argument for %s.", string), NULL);\
            }\
        } while(0)

#define CHECK_UNIQUE(array, length, string) do {\
    if (length)\
    {\
        for (int64_t i = 0; i < (length) - 1; ++i)\
        {\
            for (int64_t j = i + 1; j < length; ++j)\
            {\
                if ((array)[i] == (array)[j])\
                {\
                    return ERROR(ERROR_UNIQUE, string_create("received non-unique array %s.", string), NULL);\
                }\
            }\
        }\
    }\
} while(0)

#define CHECK_NEGATIVE_ARGUMENT(value, string) do {\
            if (value < 0)\
            {\
                return ERROR(ERROR_NULL, string_create("received negative argument for %s.", string), NULL);\
            }\
        } while(0)

#define UNUSED(x) (void)(x)

#endif
