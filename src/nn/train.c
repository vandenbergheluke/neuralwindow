#include <buffer.h>
#include <tensor.h>
#include <layer.h>
#include <train.h>
#include <optimizer.h>
#include <random.h>

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

nw_error_t *fit(uint64_t epochs,
                uint64_t number_of_samples,
                batch_t *batch,
                bool_t shuffle,
                float32_t train_split,
                float32_t valid_split,
                float32_t test_split,
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

    uint64_t iterations = number_of_samples / batch->batch_size;
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
    uint64_t test_iterations = (uint64_t) (test_split * (float32_t) iterations);
    tensor_t *y_pred = NULL;
    tensor_t *cost = NULL;

    for (uint64_t i = 0; i < epochs; ++i)
    {
        LOG("%lu / %lu epochs ", i + 1, epochs);
        for (uint64_t j = 0; j < train_iterations; ++j)
        {
            LOG("%s: %lu / %lu iterations ", dataset_type_string(TRAIN), j + 1, train_iterations);

            error = (*dataloader)(indicies[j] * batch->batch_size, batch, arguments);
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

            LOG_SCALAR_TENSOR("cost", cost);

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

            error = optimizer_step(optimizer, model);
            if (error)
            {
                return ERROR(ERROR_STEP, string_create("failed to update weights."), error);
            }
            
            LOG_NEWLINE;
        }

        with_no_gradient(true);

        uint64_t start = train_iterations;
        uint64_t end = valid_iterations + train_iterations;

        for (uint64_t j = start; j < end; ++j)
        {
            LOG("%s: %lu / %lu iterations ", dataset_type_string(VALID), j - start + 1, end - start);

            error = (*dataloader)(indicies[j] * batch->batch_size, batch, arguments);
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

            LOG_SCALAR_TENSOR("cost", cost);

            error = (*metrics)(VALID, batch->y, y_pred);
            if (error)
            {
                return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
            }

            LOG_NEWLINE;
        }

        with_no_gradient(false);
    }

    uint64_t start = train_iterations + valid_iterations;
    uint64_t end = valid_iterations + train_iterations + test_iterations;

    with_no_gradient(true);

    for (uint64_t i = start; i < end; ++i)
    {
        LOG("%s: %lu / %lu iterations ", dataset_type_string(TEST), i - start + 1, end - start);

        error = (*dataloader)(indicies[i] * batch->batch_size, batch, arguments);
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

        LOG_SCALAR_TENSOR("cost", cost);

        error = (*metrics)(TEST, batch->y, y_pred);
        if (error)
        {
            return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
        }
         
        LOG_NEWLINE;
    }

    with_no_gradient(false);

    error = (*teardown)(arguments);
    if (error)
    {
        return ERROR(ERROR_TEARDOWN, string_create("failed to teardown."), error);
    }
     
    return error;
}

string_t dataset_type_string(dataset_type_t dataset_type)
{
    switch (dataset_type)
    {
    case TRAIN:
        return "TRAIN";
    case VALID:
        return "VALID";
    case TEST:
        return "TEST";
    default:
        return "UNKNOWN_DATASET_TYPE";
    }
}