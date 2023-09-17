#ifndef TRAIN_H
#define TRAIN_H

#include <errors.h>

typedef struct model_t model_t;
typedef struct cost_t cost_t;
typedef struct dataset_t dataset_t;
typedef struct batch_t batch_t;
typedef struct optimizer_t optimizer_t;

nw_error_t *train(nw_error_t * (setup(dataset_t *)), nw_error_t *(teardown(dataset_t *)),
                  nw_error_t *(dataloader(dataset_t *, batch_t *, uint64_t)), dataset_t dataset,
                  model_t *model, cost_t *cost, optimizer_t *optimizer);
nw_error_t *inference(nw_error_t * (setup(dataset_t *)), nw_error_t *(teardown(dataset_t *)),
                      nw_error_t *(dataloader(dataset_t *, batch_t *, uint64_t)), dataset_t dataset, model_t *model);
    
#endif