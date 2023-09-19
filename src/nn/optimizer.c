#include <buffer.h>
#include <function.h>
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

    switch (datatype)
    {
    case FLOAT32:
        error = tensor_constant_float32((float32_t) optimizer->learning_rate, &learning_rate, runtime);
        break;
    case FLOAT64:
        error = tensor_constant_float64((float64_t) optimizer->learning_rate, &learning_rate, runtime);
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), error);
    }

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

    tensor_destroy(parameters->gradient);
    function_destroy(parameters->context);
    parameters->context = NULL;
    parameters->gradient = NULL;

    return error;
}

nw_error_t *update(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block)
{
    CHECK_NULL_ARGUMENT(algorithm, "algorithm");
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");

    nw_error_t *error = NULL;

    for (uint64_t i = 0; i < block->depth; ++i)
    {
        layer_t *layer = block->layers[i];
        if (!layer)
        {
            return ERROR(ERROR_NULL, string_create("failed to optimize null layer."), NULL);
        }

        transformation_type_t transformation_type = layer->transformation_type;
        transformation_t *transformation = layer->transformation;
        if (!transformation)
        {
            return ERROR(ERROR_NULL, string_create("transformation is null."), NULL);
        }

        switch (transformation_type)
        {
        case LINEAR:
            switch (algorithm_type)
            {
            case STOCASTIC_GRADIENT_DESCENT:
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, transformation->linear->weights);
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, transformation->linear->bias);
                break;
            default:
                return ERROR(ERROR_UNKNOWN_ALGORITHM, string_create("unknown algorithm %d.", (int) algorithm_type), error);
            }
            break;
        case BLOCK:
            error = update(algorithm, algorithm_type, transformation->block);
            break;
        default:
            return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown layer type %d.", transformation_type), error);
        }

        if (error)
        {
            return ERROR(ERROR_UPDATE, string_create("failed to update parameters."), error);
        }
    }

    return error;
}

nw_error_t *step(optimizer_t *optimizer, model_t *model)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    error = update(optimizer->algorithm, optimizer->algorithm_type, model->block);
    if (error)
    {
        return ERROR(ERROR_UPDATE, string_create("failed to update model parameters."), error);
    }

    return error;
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

nw_error_t *stochastic_gradient_descent_create(stochastic_gradient_descent_t **stochastic_gradient_descent,
                                               float32_t learning_rate,
                                               float32_t momentum,
                                               float32_t dampening,
                                               float32_t weight_decay,
                                               bool_t nesterov)
{
    CHECK_NULL_ARGUMENT(stochastic_gradient_descent, "stochastic_gradient_descent");

    *stochastic_gradient_descent = (stochastic_gradient_descent_t *) malloc(sizeof(stochastic_gradient_descent_t));
    if (!*stochastic_gradient_descent)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(stochastic_gradient_descent_t)), NULL);
    }

    (*stochastic_gradient_descent)->learning_rate = learning_rate;
    (*stochastic_gradient_descent)->momentum = momentum;
    (*stochastic_gradient_descent)->dampening = dampening;
    (*stochastic_gradient_descent)->weight_decay = weight_decay;
    (*stochastic_gradient_descent)->nesterov = nesterov;

    return NULL;
}

void stochastic_gradient_descent_destroy(stochastic_gradient_descent_t *stochastic_gradient_descent)
{
    if (stochastic_gradient_descent)
    {
        free(stochastic_gradient_descent);
    }
}

nw_error_t *optimizer_stochastic_gradient_descent_create(optimizer_t **optimizer,
                                                         float32_t learning_rate,
                                                         float32_t momentum,
                                                         float32_t dampening,
                                                         float32_t weight_decay,
                                                         bool_t nesterov)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");

    nw_error_t *error = NULL;
    stochastic_gradient_descent_t *stochastic_gradient_descent = NULL;
    algorithm_t *algorithm = NULL;
    algorithm_type_t algorithm_type = STOCASTIC_GRADIENT_DESCENT;

    error = stochastic_gradient_descent_create(&stochastic_gradient_descent,
                                               learning_rate,
                                               momentum,
                                               dampening,
                                               weight_decay,
                                               nesterov);
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