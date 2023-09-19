#include <buffer.h>
#include <tensor.h>
#include <layer.h>
#include <train.h>
#include <random.h>

typedef struct batch_t
{
    uint64_t batch_size;
    datatype_t datatype;
    runtime_t runtime;
    tensor_t *x;
    tensor_t *y;
} batch_t;

nw_error_t *batch_create(batch_t **batch, uint64_t batch_size, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(batch, "batch");

    *batch = (batch_t *) malloc(sizeof(batch_t));
    if (!*batch)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(batch_t)), NULL);
    }

    (*batch)->batch_size = batch_size;
    (*batch)->datatype = datatype;
    (*batch)->runtime = runtime;
    (*batch)->x = NULL;
    (*batch)->y = NULL;

    return NULL;
}

void batch_destroy(batch_t *batch)
{
    if (batch)
    {
        free(batch);
    }
}

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
                  nw_error_t *(*metrics)(dataset_type_t, const tensor_t *, const tensor_t *))
{
    nw_error_t *error = NULL;

    error = (*setup)(arguments);
    if (error)
    {
        return ERROR(ERROR_SETUP, string_create("failed to setup."), error);
    }

    uint64_t iterations = number_of_samples / batch_size;
    uint64_t indicies[iterations];

    for (uint64_t i = 0; i < iterations; ++i)
    {
        indicies[i] = i;
    }

    if (shuffle)
    {
        shuffle_array(indicies, iterations);
    }

    uint64_t train_iterations = (uint64_t) (train_split * (float32_t) iterations);
    uint64_t valid_iterations = (uint64_t) (valid_split * (float32_t) iterations);
    tensor_t *y_pred = NULL;
    tensor_t *cost = NULL;
    batch_t *batch = NULL;
    datatype_t datatype = model->datatype;
    runtime_t runtime = model->runtime;

    error = batch_create(&batch, batch_size, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create batch."), error);
    }

    for (uint64_t i = 0; i < epochs; ++i)
    {
        error = model_requires_gradient(model, true);
        if (error)
        {
            return ERROR(ERROR_REQUIRES_GRADIENT, string_create("failed to modify model's requires gradient flag."), error);
        }
        for (uint64_t j = 0; j < train_iterations; ++j)
        {
            error = (*dataloader)(indicies[j] * batch_size, batch, arguments);
            if (error)
            {
                return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
            }

            error = model_forward(model, batch->x, &y_pred);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed model forward pass."), error);
            }

            error = (*criterion)(batch->y, y_pred, &cost);
            if (error)
            {
                return ERROR(ERROR_CRITERION, string_create("failed model forward pass."), error);
            }

            error = (*metrics)(TRAIN, batch->y, y_pred);
            if (error)
            {
                return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
            }

            error = tensor_backward(cost, NULL);
            if (error)
            {
                return ERROR(ERROR_BACKWARD, string_create("failed back propogation."), error);
            }

            error = step(optimizer, model);
            if (error)
            {
                return ERROR(ERROR_STEP, string_create("failed to update weights."), error);
            }
        }

        error = model_requires_gradient(model, false);
        if (error)
        {
            return ERROR(ERROR_REQUIRES_GRADIENT, string_create("failed to modify model's requires gradient flag."), error);
        }
        for (uint64_t j = train_iterations; j < train_iterations + valid_iterations; ++j)
        {
            error = (*dataloader)(indicies[j] * batch_size, batch, arguments);
            if (error)
            {
                return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
            }

            error = model_forward(model, batch->x, &y_pred);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed model forward pass."), error);
            }

            error = criterion(batch->y, y_pred, &cost);
            if (error)
            {
                return ERROR(ERROR_CRITERION, string_create("failed model forward pass."), error);
            }

            error = (*metrics)(VALID, batch->y, y_pred);
            if (error)
            {
                return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
            }
        }
    }

    batch_destroy(batch);

    error = (*teardown)(arguments);
    if (error)
    {
        return ERROR(ERROR_TEARDOWN, string_create("failed to teardown."), error);
    }
     
    return error;
}

nw_error_t *test(uint64_t epochs,
                 uint64_t number_of_samples,
                 uint64_t batch_size,
                 model_t *model,
                 void *arguments,
                 nw_error_t *(*setup)(void *),
                 nw_error_t *(*teardown)(void *),
                 nw_error_t *(*dataloader)(uint64_t, batch_t *, void *),
                 nw_error_t *(*metrics)(dataset_type_t, const tensor_t *, const tensor_t *))
{
    nw_error_t *error = NULL;        
    batch_t *batch = NULL;
    datatype_t datatype = model->datatype;
    runtime_t runtime = model->runtime;
    tensor_t *y_pred = NULL;

    error = batch_create(&batch, batch_size, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create batch."), error);
    }

    error = model_requires_gradient(model, false);
    if (error)
    {
        return ERROR(ERROR_REQUIRES_GRADIENT, string_create("failed to modify model's requires gradient flag."), error);
    }

    for (uint64_t i = 0; i < number_of_samples / batch_size; ++i)
    {
        error = (*dataloader)(i * batch_size, batch, arguments);
        if (error)
        {
            return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
        }

        error = model_forward(model, batch->x, &y_pred);
        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed model forward pass."), error);
        }

        error = (*metrics)(TEST, batch->y, y_pred);
        if (error)
        {
            return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
        }
    }

    error = model_requires_gradient(model, true);
    if (error)
    {
        return ERROR(ERROR_REQUIRES_GRADIENT, string_create("failed to modify model's requires gradient flag."), error);
    }

    batch_destroy(batch);

    error = (*teardown)(arguments);
    if (error)
    {
        return ERROR(ERROR_TEARDOWN, string_create("failed to teardown."), error);
    }

    return error;
}