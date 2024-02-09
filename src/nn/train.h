#ifndef TRAIN_H
#define TRAIN_H

#include <errors.h>
#include <runtime.h>

typedef struct model_t model_t;
typedef struct cost_t cost_t;
typedef struct dataset_t dataset_t;
typedef struct batch_t batch_t;
typedef struct optimizer_t optimizer_t;
typedef struct tensor_t tensor_t;

#define LOG_SCALAR_TENSOR(msg, tensor) do {\
    fprintf(stdout, "%s ", msg);\
    if (!tensor || !tensor->buffer || !tensor->buffer->storage || !tensor->buffer->storage->data)\
    {\
        fprintf(stdout, "NULL");\
    }\
    else\
    {\
        switch (tensor->buffer->storage->datatype)\
        {\
        case FLOAT32:\
            fprintf(stdout, "%f ", *(float32_t *) tensor->buffer->storage->data);\
            break;\
        case FLOAT64:\
            fprintf(stdout, "%lf ", *(float64_t *) tensor->buffer->storage->data);\
            break;\
        default:\
            fprintf(stdout, " ");\
            break;\
        }\
    }\
} while(0)

#define LOG(format, ...) do {\
    fprintf(stdout, format, __VA_ARGS__);\
} while(0)

#define LOG_NEWLINE do {\
    fprintf(stdout, "\n");\
} while(0)

typedef enum dataset_type_t
{
    TRAIN,
    VALID,
    TEST
} dataset_type_t;

typedef struct batch_t
{
    int64_t batch_size;
    datatype_t datatype;
    runtime_t runtime;
    tensor_t *x;
    tensor_t *y;
} batch_t;

nw_error_t *fit(int64_t epochs,
                int64_t number_of_samples,
                batch_t *batch,
                bool_t shuffle,
                float32_t train_split,
                float32_t valid_split,
                float32_t test_split,
                model_t *model,
                optimizer_t *optimizer,
                void * arguments,
                nw_error_t *(*dataloader)(int64_t, batch_t *, void *),
                nw_error_t *(*criterion)(const tensor_t *, const tensor_t *, tensor_t **),
                nw_error_t *(*metrics)(dataset_type_t, const tensor_t *, const tensor_t *, const tensor_t *, int64_t, int64_t, int64_t, int64_t),
                nw_error_t *(*generate)(model_t *, void *, runtime_t, datatype_t),
                void *clip_gradient_norm);

nw_error_t *batch_create(batch_t **batch, int64_t batch_size, datatype_t datatype, runtime_t runtime);
void batch_destroy(batch_t *batch);
string_t dataset_type_string(dataset_type_t dataset_type);
#endif