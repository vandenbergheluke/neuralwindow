#include <buffer.h>
#include <tensor.h>
#include <layer.h>
#include <train.h>
#include <optimizer.h>
#include <random.h>
#include <graph.h>

nw_error_t *batch_create(batch_t **batch, int64_t batch_size, datatype_t datatype, runtime_t runtime)
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
                nw_error_t *(*setup)(void *),
                nw_error_t *(*teardown)(void *),
                nw_error_t *(*dataloader)(int64_t, batch_t *, void *),
                nw_error_t *(*criterion)(const tensor_t *, const tensor_t *, tensor_t **),
                nw_error_t *(*metrics)(dataset_type_t, const tensor_t *, const tensor_t *, const tensor_t *, int64_t, int64_t, int64_t, int64_t))
{
    nw_error_t *error = NULL;

    error = (*setup)(arguments);
    if (error)
    {
        return ERROR(ERROR_SETUP, string_create("failed to setup."), error);
    }

    int64_t iterations = number_of_samples / batch->batch_size;
    int64_t indicies[iterations];

    for (int64_t i = 0; i < iterations; ++i)
    {
        indicies[i] = i;
    }

    if (shuffle)
    {
        shuffle_array(indicies, iterations);
    }

    int64_t train_iterations = (int64_t) (train_split * (float32_t) iterations);
    int64_t valid_iterations = (int64_t) (valid_split * (float32_t) iterations);
    int64_t test_iterations = (int64_t) (test_split * (float32_t) iterations);
    tensor_t *y_pred = NULL;
    tensor_t *cost = NULL;

    for (int64_t i = 0; i < epochs; ++i)
    {
        for (int64_t j = 0; j < train_iterations; ++j)
        {
            error = zero_gradient_model(model);
            if (error)
            {
                return ERROR(ERROR_ZERO_GRADIENT, string_create("failed to zero gradient."), error);
            }

            error = (*dataloader)(indicies[j] * batch->batch_size, batch, arguments);
            if (error)
            {
                return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
            }

            if (!i && !j)
            {
                start_graph();
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

            if (!i && !j)
            {
                end_graph();
            }

            with_no_gradient(true);
            error = (*metrics)(TRAIN, batch->y, y_pred, cost, i + 1, epochs, j + 1, train_iterations);
            if (error)
            {
                return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
            }
            with_no_gradient(false);

            error = tensor_backward(cost, NULL);
            if (error)
            {
                return ERROR(ERROR_BACKWARD, string_create("failed back propogation."), error);
            }

            error = update_model(optimizer, model);
            if (error)
            {
                return ERROR(ERROR_STEP, string_create("failed to update weights."), error);
            }

            tensor_destroy(batch->x);
            tensor_destroy(batch->y);
            batch->x = NULL;
            batch->y = NULL;
            y_pred = NULL;
            cost = NULL;
        }

        with_no_gradient(true);

        int64_t start = train_iterations;
        int64_t end = valid_iterations + train_iterations;
        model_inference(model, true);

        for (int64_t j = start; j < end; ++j)
        {
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

            error = (*metrics)(VALID, batch->y, y_pred, cost, i + 1, epochs, j - train_iterations + 1, valid_iterations);
            if (error)
            {
                return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
            }

            tensor_destroy(batch->x);
            tensor_destroy(batch->y);
            tensor_destroy(y_pred);
            tensor_destroy(cost);
            batch->x = NULL;
            batch->y = NULL;
            y_pred = NULL;
            cost = NULL;
        }

        model_inference(model, false);
        with_no_gradient(false);
    }

    int64_t start = train_iterations + valid_iterations;
    int64_t end = valid_iterations + train_iterations + test_iterations;

    with_no_gradient(true);
    model_inference(model, true);

    for (int64_t i = start; i < end; ++i)
    {
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

        error = (*metrics)(TEST, batch->y, y_pred, cost, 1, 1, i - start + 1, test_iterations);
        if (error)
        {
            return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
        }
         
        tensor_destroy(batch->x);
        tensor_destroy(batch->y);
        tensor_destroy(y_pred);
        tensor_destroy(cost);
        batch->x = NULL;
        batch->y = NULL;
        y_pred = NULL;
        cost = NULL;
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
        return "DATASET_TYPE";
    }
}