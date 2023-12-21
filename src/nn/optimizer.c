#include <buffer.h>
#include <function.h>
#include <view.h>
#include <tensor.h>
#include <layer.h>
#include <optimizer.h>

nw_error_t *stochastic_gradient_descent(stochastic_gradient_descent_t *optimizer, tensor_t *parameters, uint64_t index)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    nw_error_t *error = NULL;
    tensor_t *learning_rate = NULL;
    tensor_t *weight_decay = NULL;
    tensor_t *weight_decay_product = NULL;
    tensor_t *parameter_update = NULL;
    tensor_t *momentum_constant = NULL;
    tensor_t *momentum_product = NULL;
    tensor_t *dampening_constant = NULL;
    tensor_t *dampening_gradient = NULL;
    tensor_t *updated_momentum = NULL;
    tensor_t *modified_momentum = NULL;
    tensor_t *nesterov_momentum = NULL;
    datatype_t datatype = parameters->buffer->storage->datatype;
    runtime_t runtime = parameters->buffer->storage->runtime;

    error = tensor_constant(optimizer->learning_rate, datatype, runtime, false, false, &learning_rate);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_constant(optimizer->weight_decay, datatype, runtime, false, false, &weight_decay);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    with_no_gradient(true);

    error = tensor_multiplication(weight_decay, parameters, &weight_decay_product);
    if (error)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
    }

    error = tensor_addition(weight_decay_product, parameters->gradient, &parameters->gradient);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }

    if (*(float32_t *) optimizer->momentum != 0.f)
    {
        if (!optimizer->momentum_buffer[index])
        {
            // iteration 0
            error = tensor_zeroes_like(parameters->gradient, &optimizer->momentum_buffer[index], false, true, true);
            if (error)
            {
                return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            }

            error = tensor_addition(optimizer->momentum_buffer[index], parameters->gradient, &optimizer->momentum_buffer[index]);
            if (error)
            {
                return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            }
        }
        else
        {
            error = tensor_constant(optimizer->momentum, datatype, runtime, false, false, &momentum_constant);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }
        
            error = tensor_multiplication(momentum_constant, optimizer->momentum_buffer[index], &momentum_product);
            if (error)
            {
                return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            }
            
            float32_t dampening_alpha = (float32_t) 1 - *(float32_t *) (optimizer->dampening);
            error = tensor_constant(&dampening_alpha, datatype, runtime, false, false, &dampening_constant);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }

            error = tensor_multiplication(dampening_constant, parameters->gradient, &dampening_gradient);
            if (error)
            {
                return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            }

            error = tensor_addition(dampening_gradient, momentum_product, &updated_momentum);
            if (error)
            {
                return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            }

            tensor_destroy(optimizer->momentum_buffer[index]);
            optimizer->momentum_buffer[index] = updated_momentum;
        }
        if (optimizer->nesterov)
        {
            if (!momentum_constant)
            {
                error = tensor_constant(optimizer->momentum, datatype, runtime, false, false, &momentum_constant);
                if (error)
                {
                    return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                }
            }
            error = tensor_multiplication(momentum_constant, optimizer->momentum_buffer[index], &modified_momentum);
            if (error)
            {
                return ERROR(ERROR_OPTIM, string_create("Nesterov momentum multiplication failed"), error);
            }

            error = tensor_addition(modified_momentum, parameters->gradient, &nesterov_momentum);
            if (error)
            {
                return ERROR(ERROR_SUBTRACTION, string_create("failed to add tensors."), error);
            }

            tensor_destroy(parameters->gradient);
            parameters->gradient = nesterov_momentum;

        }
        else
        {
            tensor_destroy(parameters->gradient);
            parameters->gradient = optimizer->momentum_buffer[index];
        }
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

    with_no_gradient(false);

    if (*(float32_t *) optimizer->momentum == 0.f)
    {
        tensor_destroy(parameters->gradient);
    }
    tensor_destroy(learning_rate);
    tensor_destroy(parameter_update);
    tensor_destroy(weight_decay);
    tensor_destroy(weight_decay_product);
    if (momentum_constant){tensor_destroy(momentum_constant);}
    if (momentum_product){tensor_destroy(momentum_product);}
    if (dampening_constant){tensor_destroy(dampening_constant);}
    if (dampening_gradient){tensor_destroy(dampening_gradient);} 
    if (modified_momentum){tensor_destroy(modified_momentum);} 
    if (nesterov_momentum){tensor_destroy(nesterov_momentum);} 
    parameters->gradient = NULL;

    return error;
}

nw_error_t *update(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block)
{
    CHECK_NULL_ARGUMENT(algorithm, "algorithm");
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");

    nw_error_t *error = NULL;
    error = update_helper(algorithm, algorithm_type, block, 0);

    return error;

}

nw_error_t *update_helper(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block, uint64_t index)
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
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, transform->linear->weights, index);
                index++;
                if (error)
                {
                    return ERROR(ERROR_UPDATE, string_create("failed stochastic gradient descent."), error);
                }
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, transform->linear->bias, index);
                index++;
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
            error = update_helper(algorithm, algorithm_type, transform->block, index);
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
                                               block_t *params,
                                               datatype_t datatype,
                                               void *learning_rate,
                                               void *momentum,
                                               void *dampening,
                                               void *weight_decay,
                                               bool_t nesterov)
{
    CHECK_NULL_ARGUMENT(stochastic_gradient_descent, "stochastic_gradient_descent");
   // CHECK_NULL_ARGUMENT(params, "params");

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

    if (*(float32_t *) (*stochastic_gradient_descent)->momentum != 0.f)
    {
        uint64_t num_params = 0;
      
        error = block_num_params(params, &num_params);
        if (error)
        {
            error = ERROR(ERROR_OPTIM, string_create("failed to count model parameters."), error);
            goto cleanup;
        }

        (*stochastic_gradient_descent)->momentum_buffer_size = num_params;

        (*stochastic_gradient_descent)->momentum_buffer = (tensor_t **) malloc(num_params * sizeof(tensor_t *));
        if (!(*stochastic_gradient_descent)->momentum_buffer)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocate momentum buffer of size %lu.",
                         (unsigned long) (num_params * sizeof(tensor_t *))), NULL);
            goto cleanup;
        }

        for (size_t i = 0; i < num_params; ++i)
        {
            (*stochastic_gradient_descent)->momentum_buffer[i] = NULL;
        }
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
        if (*(float32_t *) stochastic_gradient_descent->momentum != 0.f)
        {
            for (uint64_t i=0; i < stochastic_gradient_descent->momentum_buffer_size; ++i)
            {
                tensor_destroy(stochastic_gradient_descent->momentum_buffer[i]);
            }
            free(stochastic_gradient_descent->momentum_buffer);
        }
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
                                                         block_t *params,
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

    error = stochastic_gradient_descent_create(&stochastic_gradient_descent, params, datatype, learning_rate, momentum, dampening, weight_decay, nesterov);
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