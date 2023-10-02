#include <buffer.h>
#include <function.h>
#include <view.h>
#include <tensor.h>
#include <layer.h>
#include <optimizer.h>

nw_error_t *stochastic_gradient_descent(stochastic_gradient_descent_t *optimizer, tensor_t *parameters)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    nw_error_t *error = NULL;
    tensor_t *learning_rate = NULL;
    tensor_t *parameter_update = NULL;
    datatype_t datatype = parameters->buffer->storage->datatype;
    runtime_t runtime = parameters->buffer->storage->runtime;

    error = tensor_constant(optimizer->learning_rate, datatype, runtime, false, false, &learning_rate);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    with_no_gradient(true);

    error = tensor_multiplication(learning_rate, parameters->gradient, &parameter_update);
    if (error)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
    }

    error = tensor_subtraction(parameters, parameter_update, &parameters);
    if (error)
    {
        return ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
    }

    with_no_gradient(false);

    tensor_destroy(parameters->gradient);
    tensor_destroy(learning_rate);
    tensor_destroy(parameter_update);
    parameters->gradient = NULL;

    return error;
}

nw_error_t *update(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block)
{
    CHECK_NULL_ARGUMENT(algorithm, "algorithm");
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");

    nw_error_t *error = NULL;

    for (int64_t i = 0; i < block->depth; ++i)
    {
        layer_t *layer = block->layers[i];
        if (!layer)
        {
            return ERROR(ERROR_NULL, string_create("failed to optimize null layer."), NULL);
        }

        transform_type_t transform_type = layer->transform_type;
        transform_t *transform = layer->transform;
        if (!transform)
        {
            return ERROR(ERROR_NULL, string_create("transform is null."), NULL);
        }

        switch (transform_type)
        {
        case LINEAR:
            switch (algorithm_type)
            {
            case STOCASTIC_GRADIENT_DESCENT:
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, transform->linear->weights);
                if (error)
                {
                    return ERROR(ERROR_UPDATE, string_create("failed stochastic gradient descent."), error);
                }
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, transform->linear->bias);
                if (error)
                {
                    return ERROR(ERROR_UPDATE, string_create("failed stochastic gradient descent."), error);
                }
                break;
            default:
                return ERROR(ERROR_UNKNOWN_ALGORITHM, string_create("unknown algorithm %d.", (int) algorithm_type), error);
            }
            break;
        case BLOCK:
            error = update(algorithm, algorithm_type, transform->block);
            if (error)
            {
                return ERROR(ERROR_UPDATE, string_create("failed to update parameters."), error);
            }
            break;
        default:
            return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown layer type %d.", transform_type), error);
        }

    }

    return error;
}

nw_error_t *optimizer_step(optimizer_t *optimizer, model_t *model)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_OPTIMIZER("optimizer", optimizer);
    PRINTLN_DEBUG_MODEL("model", model);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    error = update(optimizer->algorithm, optimizer->algorithm_type, model->block);
    if (error)
    {
        return ERROR(ERROR_UPDATE, string_create("failed to update model parameters."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_MODEL("model", model);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *stochastic_gradient_descent_create(stochastic_gradient_descent_t **stochastic_gradient_descent,
                                               datatype_t datatype,
                                               void *learning_rate,
                                               void *momentum,
                                               void *dampening,
                                               void *weight_decay,
                                               bool_t nesterov)
{
    CHECK_NULL_ARGUMENT(stochastic_gradient_descent, "stochastic_gradient_descent");

    nw_error_t *error = NULL;

    *stochastic_gradient_descent = NULL;
    *stochastic_gradient_descent = (stochastic_gradient_descent_t *) malloc(sizeof(stochastic_gradient_descent_t));
    if (!*stochastic_gradient_descent)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(stochastic_gradient_descent_t)), NULL);
        goto cleanup;
    }

    (*stochastic_gradient_descent)->datatype = datatype;
    (*stochastic_gradient_descent)->learning_rate = NULL;
    (*stochastic_gradient_descent)->momentum = NULL;
    (*stochastic_gradient_descent)->dampening = NULL;
    (*stochastic_gradient_descent)->weight_decay = NULL;
    (*stochastic_gradient_descent)->nesterov = nesterov;
    size_t size = datatype_size(datatype);

    (*stochastic_gradient_descent)->learning_rate = (void *) malloc(size);
    if (!(*stochastic_gradient_descent)->learning_rate)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*stochastic_gradient_descent)->momentum = (void *) malloc(size);
    if (!(*stochastic_gradient_descent)->momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*stochastic_gradient_descent)->dampening = (void *) malloc(size);
    if (!(*stochastic_gradient_descent)->dampening)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*stochastic_gradient_descent)->weight_decay = (void *) malloc(size);
    if (!(*stochastic_gradient_descent)->weight_decay)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }
    
    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) (*stochastic_gradient_descent)->learning_rate = *(float32_t *) learning_rate;
        *(float32_t *) (*stochastic_gradient_descent)->momentum = *(float32_t *) momentum;
        *(float32_t *) (*stochastic_gradient_descent)->dampening = *(float32_t *) dampening;
        *(float32_t *) (*stochastic_gradient_descent)->weight_decay = *(float32_t *) weight_decay;
        break;
    case FLOAT64:
        *(float64_t *) (*stochastic_gradient_descent)->learning_rate = *(float64_t *) learning_rate;
        *(float64_t *) (*stochastic_gradient_descent)->momentum = *(float64_t *) momentum;
        *(float64_t *) (*stochastic_gradient_descent)->dampening = *(float64_t *) dampening;
        *(float64_t *) (*stochastic_gradient_descent)->weight_decay = *(float64_t *) weight_decay;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    stochastic_gradient_descent_destroy(*stochastic_gradient_descent);

    return error;
}

void stochastic_gradient_descent_destroy(stochastic_gradient_descent_t *stochastic_gradient_descent)
{
    if (stochastic_gradient_descent)
    {
        free(stochastic_gradient_descent->learning_rate);
        free(stochastic_gradient_descent->momentum);
        free(stochastic_gradient_descent->dampening);
        free(stochastic_gradient_descent->weight_decay);
        free(stochastic_gradient_descent);
    }
}

nw_error_t *algorithm_create(algorithm_t **algorithm, algorithm_type_t algorithm_type, void *type_algorithm)
{
    CHECK_NULL_ARGUMENT(algorithm, "algorithm");
    CHECK_NULL_ARGUMENT(type_algorithm, "type_algorithm");

    *algorithm = (algorithm_t *) malloc(sizeof(algorithm_t));
    if (!*algorithm)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(algorithm_t)), NULL);
    }

    switch (algorithm_type)
    {
    case STOCASTIC_GRADIENT_DESCENT:
        (*algorithm)->stochastic_gradient_descent = (stochastic_gradient_descent_t *) type_algorithm;
        break;
    default:
        free(*algorithm);
        return ERROR(ERROR_UNKNOWN_ALGORITHM, string_create("unknown algorithm type %d.", (int) algorithm_type), NULL);
    }

    return NULL;
}

void algorithm_destroy(algorithm_t *algorithm, algorithm_type_t algorithm_type)
{
    if (algorithm)
    {
        switch (algorithm_type)
        {
        case STOCASTIC_GRADIENT_DESCENT:
            stochastic_gradient_descent_destroy(algorithm->stochastic_gradient_descent);
            break;
        default:
            break;
        }
        free(algorithm);
    }
}

nw_error_t *optimizer_create(optimizer_t **optimizer, algorithm_t *algorithm, algorithm_type_t algorithm_type)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(algorithm, "algorithm");

    *optimizer = (optimizer_t *) malloc(sizeof(optimizer_t));
    if (!*optimizer)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(optimizer_t)), NULL);
    }

    (*optimizer)->algorithm = algorithm;
    (*optimizer)->algorithm_type = algorithm_type;
    
    return NULL;
}

void optimizer_destroy(optimizer_t *optimizer)
{
    if (optimizer)
    {
        algorithm_destroy(optimizer->algorithm, optimizer->algorithm_type);
        free(optimizer);
    }
}

string_t algorithm_type_string(algorithm_type_t algorithm_type)
{
    switch (algorithm_type)
    {
    case STOCASTIC_GRADIENT_DESCENT:
        return "STOCASTIC_GRADIENT_DESCENT";
    default:
        return "UNKNOWN_ALGORITHM";
    }
}

nw_error_t *optimizer_stochastic_gradient_descent_create(optimizer_t **optimizer,
                                                         datatype_t datatype,
                                                         void *learning_rate,
                                                         void *momentum,
                                                         void *dampening,
                                                         void *weight_decay,
                                                         bool_t nesterov)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");

    nw_error_t *error = NULL;
    stochastic_gradient_descent_t *stochastic_gradient_descent = NULL;
    algorithm_t *algorithm = NULL;
    algorithm_type_t algorithm_type = STOCASTIC_GRADIENT_DESCENT;

    error = stochastic_gradient_descent_create(&stochastic_gradient_descent, datatype, learning_rate, momentum, dampening, weight_decay, nesterov);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create stochastic gradient descent instance."), error);
    }

    error = algorithm_create(&algorithm, algorithm_type, stochastic_gradient_descent);
    if (error)
    {
        stochastic_gradient_descent_destroy(stochastic_gradient_descent);
        return ERROR(ERROR_CREATE, string_create("failed to create algorithm."), error);
    }

    error = optimizer_create(optimizer, algorithm, algorithm_type);
    if (error)
    {
        algorithm_destroy(algorithm, algorithm_type);
        return ERROR(ERROR_CREATE, string_create("failed to create optimizer."), error);
    }

    return error;
}