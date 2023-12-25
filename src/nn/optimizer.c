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
            // error = tensor_zeroes_like(parameters->gradient, &optimizer->momentum_buffer[index], false, true, true);
            // if (error)
            // {
            //     return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            // }

            // error = tensor_addition(optimizer->momentum_buffer[index], parameters->gradient, &optimizer->momentum_buffer[index]);
            // if (error)
            // {
            //     return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            // }
            error = tensor_from_data(&optimizer->momentum_buffer[index], parameters->gradient->buffer->storage->data, parameters->gradient->buffer->storage->runtime, 
                                    parameters->gradient->buffer->storage->datatype, parameters->gradient->buffer->view->rank, parameters->gradient->buffer->view->shape, 
                                    parameters->gradient->buffer->view->strides, parameters->gradient->buffer->view->offset, true, false, true);
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
            case RMS_PROP:
                error = rms_prop(algorithm->rms_prop, transform->linear->weights, index);
                index++;
                if (error)
                {
                    return ERROR(ERROR_UPDATE, string_create("failed rms prop."), error);
                }
                error = rms_prop(algorithm->rms_prop, transform->linear->bias, index);
                index++;
                if (error)
                {
                    return ERROR(ERROR_UPDATE, string_create("failed rms prop."), error);
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
    case RMS_PROP:
        (*algorithm)->rms_prop = (rms_prop_t *) type_algorithm;
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
        case RMS_PROP:
            rms_prop_destroy(algorithm->rms_prop);
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

nw_error_t *rms_prop_create(rms_prop_t **rms_prop,
                            block_t *params,
                            datatype_t datatype,
                            void *learning_rate,
                            void *momentum,
                            void *alpha,
                            void *weight_decay,
                            void *epsilon,
                            bool_t centered)
{
    CHECK_NULL_ARGUMENT(rms_prop, "rms_prop");

    nw_error_t *error = NULL;

    *rms_prop = NULL;
    *rms_prop = (rms_prop_t *)malloc(sizeof(rms_prop_t));
    if (!*rms_prop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(rms_prop_t)), NULL);
        goto cleanup;
    }

    (*rms_prop)->datatype = datatype;
    (*rms_prop)->learning_rate = NULL;
    (*rms_prop)->momentum = NULL;
    (*rms_prop)->alpha = NULL;
    (*rms_prop)->weight_decay = NULL;
    (*rms_prop)->centered = centered;
    (*rms_prop)->epsilon = NULL;
    (*rms_prop)->momentum_buffer = NULL; 

    size_t size = datatype_size(datatype);

    (*rms_prop)->learning_rate = (void *)malloc(size);
    if (!(*rms_prop)->learning_rate)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*rms_prop)->momentum = (void *)malloc(size);
    if (!(*rms_prop)->momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*rms_prop)->alpha = (void *)malloc(size);
    if (!(*rms_prop)->alpha)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*rms_prop)->weight_decay = (void *)malloc(size);
    if (!(*rms_prop)->weight_decay)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*rms_prop)->epsilon = (void *)malloc(size);
    if (!(*rms_prop)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *)(*rms_prop)->learning_rate = *(float32_t *)learning_rate;
        *(float32_t *)(*rms_prop)->momentum = *(float32_t *)momentum;
        *(float32_t *)(*rms_prop)->alpha = *(float32_t *)alpha;
        *(float32_t *)(*rms_prop)->weight_decay = *(float32_t *)weight_decay;
        *(float32_t *)(*rms_prop)->epsilon = *(float32_t *)epsilon;
        break;
    case FLOAT64:
        *(float64_t *)(*rms_prop)->learning_rate = *(float64_t *)learning_rate;
        *(float64_t *)(*rms_prop)->momentum = *(float64_t *)momentum;
        *(float64_t *)(*rms_prop)->alpha = *(float64_t *)alpha;
        *(float64_t *)(*rms_prop)->weight_decay = *(float64_t *)weight_decay;
        *(float64_t *)(*rms_prop)->epsilon = *(float64_t *)epsilon;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
        goto cleanup;
    }
    
    uint64_t num_params = 0;
      
    error = block_num_params(params, &num_params);
    if (error)
    {
        error = ERROR(ERROR_OPTIM, string_create("failed to count model parameters."), error);
        goto cleanup;
    }
    (*rms_prop)->buffer_size = num_params;

    (*rms_prop)->square_average = (tensor_t **)malloc(num_params * sizeof(tensor_t *));
    if (!(*rms_prop)->square_average)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION,
                        string_create("failed to allocate square average buffer of size %lu.",
                                    (unsigned long)(num_params * sizeof(tensor_t *))),
                        NULL);
        goto cleanup;
    }

    (*rms_prop)->average_gradient = (tensor_t **)malloc(num_params * sizeof(tensor_t *));
    if (!(*rms_prop)->average_gradient)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION,
                        string_create("failed to allocate average gradient buffer of size %lu.",
                                    (unsigned long)(num_params * sizeof(tensor_t *))),
                        NULL);
        goto cleanup;
    }

    error = initialize_zero_buffer(params, (*rms_prop)->square_average, 0);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create square average buffer tensor for null layer."), NULL);
        goto cleanup;
    }

    error = initialize_zero_buffer(params, (*rms_prop)->average_gradient, 0);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create average gradient buffer tensor for null layer."), NULL);
        goto cleanup;
    }

    if (*(float32_t *)(*rms_prop)->momentum != 0.f)
    {
        (*rms_prop)->momentum_buffer = (tensor_t **)malloc(num_params * sizeof(tensor_t *));
        if (!(*rms_prop)->momentum_buffer)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION,
                          string_create("failed to allocate momentum buffer of size %lu.",
                                        (unsigned long)(num_params * sizeof(tensor_t *))),
                          NULL);
            goto cleanup;
        }

        error = initialize_zero_buffer(params, (*rms_prop)->momentum_buffer, 0);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create momentum buffer tensor for null layer."), NULL);
            goto cleanup;
        }
    }

    return error;

cleanup:
    rms_prop_destroy(*rms_prop);
    return error;
}

void rms_prop_destroy(rms_prop_t *rms_prop)
{
    if (rms_prop)
    {
        if (*(float32_t *) rms_prop->momentum != 0.f)
        {
            for (uint64_t i=0; i < rms_prop->buffer_size; ++i)
            {
                tensor_destroy(rms_prop->momentum_buffer[i]);
            }
            free(rms_prop->momentum_buffer);
        }

        for (uint64_t i=0; i < rms_prop->buffer_size; ++i)
        {
            tensor_destroy(rms_prop->square_average[i]);
            tensor_destroy(rms_prop->average_gradient[i]);
        }
        free(rms_prop->square_average);
        free(rms_prop->average_gradient);

        free(rms_prop->learning_rate);
        free(rms_prop->momentum);
        free(rms_prop->alpha);
        free(rms_prop->weight_decay);
        free(rms_prop->epsilon);
        free(rms_prop);
    }
}


const nw_error_t *initialize_zero_buffer(block_t *param, tensor_t **buffer, uint64_t index)
{
    nw_error_t *error = NULL;

    for (uint64_t i = 0; i < param->depth; ++i)
    {
        layer_t *layer = param->layers[i];
        if (!layer)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create buffer tensor for null layer."), NULL);
        }

        transform_type_t transform_type = layer->transform_type;
        transform_t *transform = layer->transform;
        if (!transform)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create buffer tensor, transform is null."), NULL);
        }

        switch (transform_type)
        {
        case LINEAR:
            tensor_zeroes_like(transform->linear->weights, &buffer[index], false, true, true);
            index++;
            tensor_zeroes_like(transform->linear->bias, &buffer[index], false, true, true);
            index++;
            break;
        case BLOCK:
            error = initialize_zero_buffer(transform->block, buffer, index);
            if (error)
            {
                return ERROR(ERROR_UPDATE, string_create("failed to initilize buffer for block."), error);
            }
            break;
        default:
            return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown layer type %d.", transform_type), error);
        }
    }
    return error;
}

nw_error_t *optimizer_rms_prop_create(optimizer_t **optimizer,
                                        block_t *params,
                                        datatype_t datatype,
                                        void *learning_rate,
                                        void *momentum,
                                        void *alpha,
                                        void *weight_decay,
                                        void *epsilon,
                                        bool_t centered)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");

    nw_error_t *error = NULL;
    rms_prop_t *rms_prop = NULL;
    algorithm_t *algorithm = NULL;
    algorithm_type_t algorithm_type = RMS_PROP;

    error = rms_prop_create(&rms_prop, params, datatype, learning_rate, momentum, alpha, weight_decay, epsilon, centered);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create rms prop instance."), error);
    }

    error = algorithm_create(&algorithm, algorithm_type, rms_prop);
    if (error)
    {
        rms_prop_destroy(rms_prop);
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

nw_error_t *rms_prop(rms_prop_t *optimizer, tensor_t *parameters, uint64_t index)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    nw_error_t *error = NULL;
    tensor_t *learning_rate = NULL;
    tensor_t *weight_decay = NULL;
    tensor_t *weight_decay_product = NULL;
    tensor_t *temp_gradient_addition = NULL;
    tensor_t *alpha_constant = NULL;
    tensor_t *alpha_product = NULL;
    tensor_t *one_minus_alpha_constant = NULL;
    tensor_t *squared_current_gradient = NULL;
    tensor_t *one_minus_alpha_product = NULL;
    tensor_t *square_average_telda = NULL;
    tensor_t *temp_optimizer_square_average = NULL;
    tensor_t *learning_rate_gradient = NULL;
    tensor_t *square_average_telda_root = NULL;
    tensor_t *epsilon_constant = NULL;
    tensor_t *square_average_telda_epsilon = NULL;
    tensor_t *parameter_update = NULL;
    tensor_t *momentum_const_buffer = NULL;
    tensor_t *temp_gradient = NULL;
    tensor_t *momentum_constant = NULL;
    tensor_t *momentum_product = NULL;
    tensor_t *updated_momentum = NULL;
    tensor_t *modified_momentum = NULL;
    tensor_t *centered_grad = NULL;
    tensor_t *alpha_average_grad = NULL;
    tensor_t *average_gradient_squared = NULL;
    tensor_t *updated_average_grad = NULL;
   
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

    error = tensor_addition(weight_decay_product, parameters->gradient, &temp_gradient_addition);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }
    tensor_destroy(parameters->gradient);
    parameters->gradient = temp_gradient_addition;

    error = tensor_constant(optimizer->alpha, datatype, runtime, false, false, &alpha_constant);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_multiplication(alpha_constant, optimizer->square_average[index], &alpha_product);
    if (error)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
    }

    switch(datatype)
    {
        case FLOAT32:
            float32_t alpha_float32 = (float32_t) 1 - *(float32_t *) (optimizer->alpha);
            error = tensor_constant(&alpha_float32, datatype, runtime, false, false, &one_minus_alpha_constant);
            break;
        case FLOAT64:
            float64_t alpha_float64 = (float64_t) 1 - *(float64_t *) (optimizer->alpha);
            error = tensor_constant(&alpha_float64, datatype, runtime, false, false, &one_minus_alpha_constant);
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
            goto cleanup;
    }
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }
    
    error = tensor_multiplication(parameters->gradient, parameters->gradient, &squared_current_gradient);
    if (error)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
    }

    error = tensor_multiplication(one_minus_alpha_constant, squared_current_gradient, &one_minus_alpha_product);
    if (error)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
    }

    error = tensor_addition(alpha_product, one_minus_alpha_product, &square_average_telda);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }

    tensor_destroy(optimizer->square_average[index]); 
    error = tensor_zeroes_like(square_average_telda, &optimizer->square_average[index], false, true, true);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }

    error = tensor_addition(optimizer->square_average[index], square_average_telda, &temp_optimizer_square_average);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }
    tensor_destroy(optimizer->square_average[index]);
    optimizer->square_average[index] = temp_optimizer_square_average;

    if (optimizer->centered)
    {
        error = tensor_multiplication(one_minus_alpha_constant, parameters->gradient, &centered_grad);
        if (error)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        }

        error = tensor_multiplication(optimizer->average_gradient[index], alpha_constant, &alpha_average_grad);
        if (error)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        }

        error = tensor_addition(alpha_average_grad, centered_grad, &updated_average_grad);
        if (error)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        }
        tensor_destroy(optimizer->average_gradient[index]);
        optimizer->average_gradient[index] = updated_average_grad;

        error = tensor_multiplication(optimizer->average_gradient[index], optimizer->average_gradient[index], &average_gradient_squared);
        if (error)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        }

        error = tensor_subtraction(square_average_telda, average_gradient_squared, &square_average_telda);
        if (error)
        {
            return ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        }
    }

    error = tensor_square_root(square_average_telda, &square_average_telda_root);
    if (error)
    {
        return ERROR(ERROR_SQUARE_ROOT, string_create("failed to perfrom square root on tensor."), error);
    }
    
    error = tensor_constant(optimizer->epsilon, datatype, runtime, false, false, &epsilon_constant);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_addition(square_average_telda_root, epsilon_constant, &square_average_telda_epsilon);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }

    if (*(float32_t *) optimizer->momentum != 0.f)
    {
        error = tensor_division(parameters->gradient, square_average_telda_epsilon, &temp_gradient);
        if (error)
        {
            return ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        }

        error = tensor_constant(optimizer->momentum, datatype, runtime, false, false, &momentum_constant);
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        }

        error = tensor_multiplication(momentum_constant, optimizer->momentum_buffer[index], &momentum_const_buffer);
        if (error)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        }

        error = tensor_addition(momentum_const_buffer, temp_gradient, &updated_momentum);
        if (error)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        }
        tensor_destroy(optimizer->momentum_buffer[index]);
        optimizer->momentum_buffer[index] = updated_momentum;

        error = tensor_multiplication(learning_rate, updated_momentum, &parameter_update);
        if (error)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        }
    }
    else
    {
        error = tensor_multiplication(learning_rate, parameters->gradient, &learning_rate_gradient);
        if (error)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        }

        error = tensor_division(learning_rate_gradient, square_average_telda_epsilon, &parameter_update);
        if (error)
        {
            return ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        }
    }

    error = tensor_subtraction(parameters, parameter_update, &parameters);
    if (error)
    {
        return ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
    }

    with_no_gradient(false);
    tensor_destroy(learning_rate);
    tensor_destroy(weight_decay);
    tensor_destroy(weight_decay_product);
    tensor_destroy(temp_gradient_addition);
    tensor_destroy(alpha_constant);
    tensor_destroy(alpha_product);
    tensor_destroy(one_minus_alpha_constant);
    tensor_destroy(squared_current_gradient);
    tensor_destroy(one_minus_alpha_product);
    tensor_destroy(square_average_telda);
    tensor_destroy(learning_rate_gradient);
    tensor_destroy(square_average_telda_root);
    tensor_destroy(epsilon_constant);
    tensor_destroy(square_average_telda_epsilon);
    tensor_destroy(parameter_update);
    if (*(float32_t *) optimizer->momentum != 0.f)
    {
        tensor_destroy(momentum_const_buffer);
        tensor_destroy(temp_gradient);
        tensor_destroy(momentum_constant);
    }
    if (optimizer->centered)
    {
        tensor_destroy(centered_grad);
        tensor_destroy(alpha_average_grad);
        tensor_destroy(average_gradient_squared);
    }
    parameters->gradient = NULL;
    return error;

cleanup:
    // if (*(float32_t *) optimizer->momentum == 0.f)
    // {
    //     tensor_destroy(parameters->gradient);
    // }
    // tensor_destroy(learning_rate);
    // tensor_destroy(parameter_update);
    // tensor_destroy(weight_decay);
    // tensor_destroy(weight_decay_product);
    // if (momentum_constant){tensor_destroy(momentum_constant);}
    // if (momentum_product){tensor_destroy(momentum_product);}
    // if (dampening_constant){tensor_destroy(dampening_constant);}
    // if (dampening_gradient){tensor_destroy(dampening_gradient);} 
    // if (modified_momentum){tensor_destroy(modified_momentum);}
    parameters->gradient = NULL;
    return error;
}