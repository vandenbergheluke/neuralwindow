#ifndef TRAIN_H
#define TRAIN_H

#include <errors.h>

typedef struct model_t model_t;
typedef struct cost_t cost_t;
typedef struct dataset_t dataset_t;
typedef struct batch_t batch_t;
typedef struct optimizer_t optimizer_t;

typedef enum dataset_type_t
{
    TRAIN,
    VALID,
    TEST
} dataset_type_t;

typedef struct batch_t
{
    uint64_t batch_size;
    datatype_t datatype;
    runtime_t runtime;
    tensor_t *x;
    tensor_t *y;
} batch_t;

nw_error_t *train(uint64_t epochs,
                  uint64_t number_of_samples,
                  uint64_t batch_size,
                  bool_t shuffle,
                  float32_t train_split,
                  float32_t valid_split,
                  model_t *model,
                  optimizer_t *optimizer,
                  void * arguments,
                  nw_error_t *(*setup)(void *),
                  nw_error_t *(*teardown)(void *),
                  nw_error_t *(*dataloader)(uint64_t, batch_t *, void *),
                  nw_error_t *(*criterion)(const tensor_t *, const tensor_t *, tensor_t **),
                  nw_error_t *(*metrics)(dataset_type_t, const tensor_t *, const tensor_t *));

nw_error_t *test(uint64_t epochs,
                 uint64_t number_of_samples,
                 uint64_t batch_size,
                 model_t *model,
                 void * arguments,
                 nw_error_t *(*setup)(void *),
                 nw_error_t *(*teardown)(void *),
                 nw_error_t *(*dataloader)(uint64_t, batch_t *, void *),
                 nw_error_t *(*metrics)(dataset_type_t, const tensor_t *, const tensor_t *));
#endif