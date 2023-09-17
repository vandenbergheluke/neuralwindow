#include <data.h>
#include <train.h>
#include <random.h>


nw_error_t *train(nw_error_t * (setup(dataset_t *)), nw_error_t *(teardown(dataset_t *)),
                  nw_error_t *(dataloader(dataset_t *, batch_t *, uint64_t)), dataset_t *dataset,
                  model_t *model, nw_error_t *(criterion(tensor_t *, tensor_t *, tensor_t **)), optimizer_t *optimizer,
                  uint64_t epochs, nw_error_t *(metrics(dataset_type_t, tensor_t *, tensor_t *, tensor_t *)))
{
    nw_error_t *error = NULL;

    error = setup(dataset);
    if (error)
    {
        return ERROR(ERROR_SETUP, string_create("failed to setup."), error);
    }

    uint64_t iterations = dataset->number_of_samples / dataset->batch_size;
    if (iterations % dataset->batch_size)
    {
        --iterations;
    }

    uint64_t indicies[iterations];

    for (uint64_t i = 0; i < iterations; ++i)
    {
        indicies[i] = i;
    }

    if (dataset->shuffle)
    {
        shuffle(indicies, iterations);
    }

    uint64_t train_iterations = dataset->train_split * iterations;
    uint64_t valid_iterations = dataset->valid_split * iterations;
    tensor_t *x = NULL;
    tensor_t *y_true = NULL;
    tensor_t *y_pred = NULL;
    tensor_t *cost = NULL;
    batch_t batch = (batch_t) { .x = NULL, .y = NULL };

    for (uint64_t i = 0; i < epochs; ++i)
    {
        for (uint64_t j = 0; j < train_iterations; ++j)
        {
            error = dataloader(dataset, &batch, indicies[j] * dataset->batch_size);
            if (error)
            {
                return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
            }

            x = batch.x;
            y_true = batch.y;

            error = forward(model, x, &y_pred, true);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed model forward pass."), error);
            }

            error = criterion(y_true, y_pred, &cost);
            if (error)
            {
                return ERROR(ERROR_CRITERION, string_create("failed model forward pass."), error);
            }

            error = metrics(TRAIN, y_true, y_pred, cost);
            if (error)
            {
                return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
            }

            error = backward(model, cost);
            if (error)
            {
                return ERROR(ERROR_BACKWARD, string_create("failed back propogation."), error);
            }

            error = step(optimizer, model);
            if (error)
            {
                return ERROR(ERROR_STEP, string_create("failed to update weights."), error);
            }

            error = reset_gradients(model);
            if (error)
            {
                return ERROR(ERROR_RESET, string_create("failed to reset gradients."), error);
            }
        }

        for (uint64_t j = train_iterations; j < train_iterations + valid_iterations; ++j)
        {
            error = dataloader(dataset, &batch, indicies[j] * dataset->batch_size);
            if (error)
            {
                return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
            }

            x = batch.x;
            y_true = batch.y;

            error = forward(model, x, &y_pred, false);

            error = criterion(y_true, y_pred, &cost);

            error = metrics(VALID, y_true, y_pred, cost);
        }
    }

    error = teardown(dataset);
    if (error)
    {
        return ERROR(ERROR_TEARDOWN, string_create("failed to teardown."), error);
    }
     
    return error;
}

nw_error_t *inference(nw_error_t * (setup(dataset_t *)), nw_error_t *(teardown(dataset_t *)),
                       nw_error_t *(dataloader(dataset_t *, batch_t *, uint64_t)), dataset_t *dataset, model_t *model);