#include <buffer.h>
#include <function.h>
#include <view.h>
#include <tensor.h>
#include <layer.h>
#include <optimizer.h>
#include <math.h>

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
    case RMS_PROP:
        (*algorithm)->rms_prop = (rms_prop_t *) type_algorithm;
        break;
    case ADAM:
        (*algorithm)->adam = (adam_t *) type_algorithm;
        break;
    default:
        free(*algorithm);
        return ERROR(ERROR_ALGORITHM, string_create("unknown algorithm type %d.", (int) algorithm_type), NULL);
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
        case ADAM:
            adam_destroy(algorithm->adam);
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
    case RMS_PROP:
        return "RMS_PROP";
    case ADAM:
        return "ADAM";
    default:
        return "ALGORITHM";
    }
}

nw_error_t *optimizer_stochastic_gradient_descent_create(optimizer_t **optimizer, datatype_t datatype, void *learning_rate,
                                                         void *momentum, void *dampening, void *weight_decay, bool_t nesterov)
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

nw_error_t *optimizer_rms_prop_create(optimizer_t **optimizer, datatype_t datatype, void *learning_rate,
                                      void *momentum, void *alpha, void *weight_decay, void *epsilon, bool_t centered)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");

    nw_error_t *error = NULL;
    rms_prop_t *rms_prop = NULL;
    algorithm_t *algorithm = NULL;
    algorithm_type_t algorithm_type = RMS_PROP;

    error = rms_prop_create(&rms_prop, datatype, learning_rate, momentum, alpha, weight_decay, epsilon, centered);
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

nw_error_t *optimizer_adam_create(optimizer_t **optimizer, datatype_t datatype, void *learning_rate,
                                  void *beta_1, void *beta_2, void *weight_decay, void *epsilon)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");

    nw_error_t *error = NULL;
    adam_t *adam = NULL;
    algorithm_t *algorithm = NULL;
    algorithm_type_t algorithm_type = ADAM;

    error = adam_create(&adam, datatype, learning_rate, beta_1, beta_2, weight_decay, epsilon);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create adam instance."), error);
    }

    error = algorithm_create(&algorithm, algorithm_type, adam);
    if (error)
    {
        adam_destroy(adam);
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

nw_error_t *stochastic_gradient_descent_create(stochastic_gradient_descent_t **stochastic_gradient_descent, datatype_t datatype,
                                               void *learning_rate, void *momentum, void *dampening, void *weight_decay, bool_t nesterov)
{
    CHECK_NULL_ARGUMENT(stochastic_gradient_descent, "stochastic_gradient_descent");
    CHECK_NULL_ARGUMENT(learning_rate, "learning_rate");
    CHECK_NULL_ARGUMENT(momentum, "momentum");
    CHECK_NULL_ARGUMENT(dampening, "dampening");
    CHECK_NULL_ARGUMENT(weight_decay, "weight_decay");

    nw_error_t *error = NULL;

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
    (*stochastic_gradient_descent)->momentum_buffer = NULL;
    
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

    error = map_create(&(*stochastic_gradient_descent)->momentum_buffer);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map"), NULL);
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
        for (uint64_t i = 0; i < stochastic_gradient_descent->momentum_buffer->capacity; ++i)
        {
            if (stochastic_gradient_descent->momentum_buffer->entries[i].data)
            {
                tensor_destroy((tensor_t *) stochastic_gradient_descent->momentum_buffer->entries[i].data);
            }
        }
        map_destroy(stochastic_gradient_descent->momentum_buffer);
        free(stochastic_gradient_descent->learning_rate);
        free(stochastic_gradient_descent->momentum);
        free(stochastic_gradient_descent->dampening);
        free(stochastic_gradient_descent->weight_decay);
        free(stochastic_gradient_descent);
    }
}

nw_error_t *rms_prop_create(rms_prop_t **rms_prop, datatype_t datatype, void *learning_rate,
                            void *momentum, void *alpha, void *weight_decay, void *epsilon, bool_t centered)
{
    CHECK_NULL_ARGUMENT(rms_prop, "rms_prop");
    CHECK_NULL_ARGUMENT(learning_rate, "learning_rate");
    CHECK_NULL_ARGUMENT(momentum, "momentum");
    CHECK_NULL_ARGUMENT(alpha, "alpha");
    CHECK_NULL_ARGUMENT(weight_decay, "weight_decay");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;

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
    
    error = map_create(&(*rms_prop)->momentum_buffer);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), NULL);
        goto cleanup;
    }

    error = map_create(&(*rms_prop)->square_average);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), NULL);
        goto cleanup;
    }

    error = map_create(&(*rms_prop)->average_gradient);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), NULL);
        goto cleanup;
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
        map_t *maps[] = {rms_prop->momentum_buffer, rms_prop->square_average, rms_prop->average_gradient};
        for (int i = 0; i < 3; ++i)
        {
            for (uint64_t j = 0; j < maps[i]->capacity; ++j)
            {
                if (maps[i]->entries[j].data)
                {
                    tensor_destroy((tensor_t *) maps[i]->entries[j].data);
                }
            }
            map_destroy(maps[i]);
        }
        free(rms_prop->learning_rate);
        free(rms_prop->momentum);
        free(rms_prop->alpha);
        free(rms_prop->weight_decay);
        free(rms_prop->epsilon);
        free(rms_prop);
    }
}

nw_error_t *adam_create(adam_t **adam, datatype_t datatype, void *learning_rate,
                        void *beta_1, void *beta_2, void *weight_decay, void *epsilon)
{
    CHECK_NULL_ARGUMENT(adam, "adam");

    nw_error_t *error = NULL;

    *adam = (adam_t *)malloc(sizeof(adam_t));
    if (!*adam)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(adam_t)), NULL);
        goto cleanup;
    }

    (*adam)->datatype = datatype;
    (*adam)->learning_rate = NULL;
    (*adam)->beta_1 = NULL;
    (*adam)->beta_2 = NULL;
    (*adam)->weight_decay = NULL;
    (*adam)->epsilon = NULL;
    (*adam)->iteration = NULL;
    (*adam)->first_moment = NULL;
    (*adam)->second_moment = NULL;

    size_t size = datatype_size(datatype);

    (*adam)->learning_rate = (void *)malloc(size);
    if (!(*adam)->learning_rate)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*adam)->beta_1 = (void *)malloc(size);
    if (!(*adam)->beta_1)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*adam)->beta_2 = (void *)malloc(size);
    if (!(*adam)->beta_2)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*adam)->weight_decay = (void *)malloc(size);
    if (!(*adam)->weight_decay)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    (*adam)->epsilon = (void *)malloc(size);
    if (!(*adam)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *)(*adam)->learning_rate = *(float32_t *)learning_rate;
        *(float32_t *)(*adam)->beta_1 = *(float32_t *)beta_1;
        *(float32_t *)(*adam)->beta_2 = *(float32_t *)beta_2;
        *(float32_t *)(*adam)->weight_decay = *(float32_t *)weight_decay;
        *(float32_t *)(*adam)->epsilon = *(float32_t *)epsilon;
        break;
    case FLOAT64:
        *(float64_t *)(*adam)->learning_rate = *(float64_t *)learning_rate;
        *(float64_t *)(*adam)->beta_1 = *(float64_t *)beta_1;
        *(float64_t *)(*adam)->beta_2 = *(float64_t *)beta_2;
        *(float64_t *)(*adam)->weight_decay = *(float64_t *)weight_decay;
        *(float64_t *)(*adam)->epsilon = *(float64_t *)epsilon;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
        goto cleanup;
    }

    error = map_create(&(*adam)->first_moment);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), NULL);
        goto cleanup;
    }

    error = map_create(&(*adam)->second_moment);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), NULL);
        goto cleanup;
    }

    error = map_create(&(*adam)->iteration);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), NULL);
        goto cleanup;
    }

    return error;

cleanup:
    adam_destroy(*adam);
    return error;
}

void adam_destroy(adam_t *adam)
{
    if (adam)
    {
        map_t *maps[] = {adam->iteration, adam->first_moment, adam->second_moment};
        for (int i = 0; i < 3; ++i)
        {
            for (uint64_t j = 0; j < maps[i]->capacity; ++j)
            {
                if (!i)
                {
                    free(maps[i]->entries[j].data);
                }
                else
                {
                    if (maps[i]->entries[j].data)
                    {
                        tensor_destroy((tensor_t *) maps[i]->entries[j].data);
                    }
                }
            }
            map_destroy(maps[i]);
        }
        free(adam->learning_rate);
        free(adam->beta_1);
        free(adam->beta_2);
        free(adam->weight_decay);
        free(adam->epsilon);
        free(adam);
    }
    return;
}

nw_error_t *optimizer_step(optimizer_t *optimizer, model_t *model)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_OPTIMIZER("optimizer", optimizer);
    PRINTLN_DEBUG_MODEL("model", model);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = update(optimizer->algorithm, optimizer->algorithm_type, model->block);
    if (error)
    {
        return ERROR(ERROR_UPDATE, string_create("failed to update model parameters."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_MODEL("model", model);
    PRINT_DEBUG_NEWLINE;

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
            return ERROR(ERROR_NULL, string_create("failed to update null layer."), NULL);
        }

        transform_type_t transform_type = layer->transform_type;
        transform_t *transform = layer->transform;
        if (!transform)
        {
            return ERROR(ERROR_NULL, string_create("transform is null."), NULL);
        }

        int number_parameters = 2;
        tensor_t *parameters[number_parameters];
        switch (transform_type)
        {
        case LINEAR:
            parameters[0] = transform->linear->weights;
            parameters[1] = transform->linear->bias;
            break;
        case CONVOLUTION_2D:
        case CONVOLUTION_TRANSPOSE_2D:
            parameters[0] = transform->convolution_2d->kernel;
            parameters[1] = transform->convolution_2d->bias;
            break;
        case DROPOUT:
            continue;
        case BLOCK:
            error = update(algorithm, algorithm_type, transform->block);
            if (error)
            {
                return ERROR(ERROR_UPDATE, string_create("failed to update parameters."), error);
            }
            continue;
        default:
            return ERROR(ERROR_LAYER_TYPE, string_create("unknown layer type %d.", (int) transform_type), error);
        }

        for (int j = 0; j < number_parameters; ++j)
        {
            switch (algorithm_type)
            {
            case STOCASTIC_GRADIENT_DESCENT:
                error = stochastic_gradient_descent(algorithm->stochastic_gradient_descent, parameters[j]);
                break;
            case RMS_PROP:
                error = rms_prop(algorithm->rms_prop, parameters[j]);
                break;
            case ADAM:
                error = adam(algorithm->adam, parameters[j]);
                break;
            default:
                return ERROR(ERROR_ALGORITHM, string_create("unknown algorithm %d.", (int) algorithm_type), error);
            }

            if (error)
            {
                return ERROR(ERROR_UPDATE, string_create("failed update parameters."), error);
            }

            // Zero gradient 
            tensor_destroy(parameters[j]->gradient);
            parameters[j]->gradient = NULL;
        }
    }
    return error;
}

static bool_t is_zero(void *value, datatype_t datatype)
{
    if (!value)
    {
        return true;
    }

    switch (datatype)
    {
    case FLOAT32:
        return fabsf(*(float32_t *) value) < FLT_EPSILON;
        break;
    case FLOAT64:
        return fabs(*(float64_t *) value) < DBL_EPSILON;
        break;
    }

    return false;
}

nw_error_t *stochastic_gradient_descent(stochastic_gradient_descent_t *optimizer, tensor_t *parameters)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("parameters", parameters);

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
    tensor_t *initial_momentum = NULL;
    tensor_t *nesterov_momentum = NULL;
    tensor_t *weight_decay_sum = NULL;
    string_t key = string_create("%lu", parameters->id);
    datatype_t datatype = parameters->buffer->storage->datatype;
    runtime_t runtime = parameters->buffer->storage->runtime;

    with_no_gradient(true);

    if (!is_zero(optimizer->weight_decay, optimizer->datatype))
    {
        error = tensor_constant(optimizer->weight_decay, datatype, runtime, false, false, &weight_decay);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(weight_decay, parameters, &weight_decay_product);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(weight_decay_product, parameters->gradient, &weight_decay_sum);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }

        tensor_destroy(parameters->gradient);
        parameters->gradient = NULL;
        error = tensor_as_tensor(weight_decay_sum, &parameters->gradient);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    if (!is_zero(optimizer->momentum, optimizer->datatype))
    {
        if (!map_contains(optimizer->momentum_buffer, key))
        {
            error = tensor_as_tensor(parameters->gradient, &updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }

            error = map_set(optimizer->momentum_buffer, key, updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_SET, string_create("failed to set map entry."), error);
                goto cleanup;
            }
        }
        else
        {
            error = tensor_constant(optimizer->momentum, datatype, runtime, false, false, &momentum_constant);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }

            error = map_get(optimizer->momentum_buffer, key, (void **) &initial_momentum);
            if (error)
            {
                error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
                goto cleanup;
            }

            error = tensor_multiplication(momentum_constant, initial_momentum, &momentum_product);
            if (error)
            {
                error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
                goto cleanup;
            }

            switch(datatype)
            {
                case FLOAT32:
                    float32_t dampening_alpha_32 = (float32_t) 1 - *(float32_t *) (optimizer->dampening);
                    error = tensor_constant(&dampening_alpha_32, datatype, runtime, false, false, &dampening_constant);
                    break;
                case FLOAT64:
                    float64_t dampening_alpha_64 = (float64_t) 1 - *(float64_t *) (optimizer->dampening);
                    error = tensor_constant(&dampening_alpha_64, datatype, runtime, false, false, &dampening_constant);
                    break;
                default:
                    error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
                    break;
            }
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }

            error = tensor_multiplication(dampening_constant, parameters->gradient, &dampening_gradient);
            if (error)
            {
                error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
                goto cleanup;
            }

            error = tensor_addition(dampening_gradient, momentum_product, &updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
                goto cleanup;
            }

            tensor_destroy(initial_momentum);
            initial_momentum = NULL;

            error = map_set(optimizer->momentum_buffer, key, updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_SET, string_create("failed to set entry."), error);
                goto cleanup;
            }
        }
        if (optimizer->nesterov)
        {
            if (!momentum_constant)
            {
                error = tensor_constant(optimizer->momentum, datatype, runtime, false, false, &momentum_constant);
                if (error)
                {
                    error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                    goto cleanup;
                }
            }
            error = tensor_multiplication(momentum_constant, updated_momentum, &modified_momentum);
            if (error)
            {
                error = ERROR(ERROR_MULTIPLICATION, string_create("Nesterov momentum multiplication failed"), error);
                goto cleanup;
            }

            error = tensor_addition(modified_momentum, parameters->gradient, &nesterov_momentum);
            if (error)
            {
                error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
                goto cleanup;
            }

            tensor_destroy(parameters->gradient);
            parameters->gradient = NULL;
            error = tensor_as_tensor(nesterov_momentum, &parameters->gradient);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            tensor_destroy(parameters->gradient);
            parameters->gradient = NULL;
            error = tensor_as_tensor(updated_momentum, &parameters->gradient);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }
        }
    }

    error = tensor_constant(optimizer->learning_rate, datatype, runtime, false, false, &learning_rate);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }
    
    error = tensor_multiplication(learning_rate, parameters->gradient, &parameter_update);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_subtraction(parameters, parameter_update, &parameters);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    with_no_gradient(false);

cleanup:
    string_destroy(key);
    tensor_destroy(learning_rate);
    tensor_destroy(weight_decay);
    tensor_destroy(weight_decay_product);
    tensor_destroy(parameter_update);
    tensor_destroy(momentum_constant);
    tensor_destroy(momentum_product);
    tensor_destroy(dampening_constant);
    tensor_destroy(dampening_gradient);
    tensor_destroy(modified_momentum);
    tensor_destroy(initial_momentum);
    tensor_destroy(nesterov_momentum);
    tensor_destroy(weight_decay_sum);
    return error;
}

nw_error_t *rms_prop(rms_prop_t *optimizer, tensor_t *parameters)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    nw_error_t *error = NULL;
    tensor_t *learning_rate = NULL;
    tensor_t *weight_decay = NULL;
    tensor_t *weight_decay_product = NULL;
    tensor_t *alpha_constant = NULL;
    tensor_t *alpha_product = NULL;
    tensor_t *one_minus_alpha_constant = NULL;
    tensor_t *squared_current_gradient = NULL;
    tensor_t *square_average = NULL;
    tensor_t *square_average_initial = NULL;
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
    tensor_t *average_gradient = NULL;
    tensor_t *average_gradient_initial = NULL;
    tensor_t *updated_average_grad = NULL;
    tensor_t *weight_decay_sum = NULL;
    tensor_t *initial_momentum = NULL;
    datatype_t datatype = parameters->buffer->storage->datatype;
    runtime_t runtime = parameters->buffer->storage->runtime;
    string_t key = string_create("%lu", parameters->id);

    with_no_gradient(true);

    if (!is_zero(optimizer->weight_decay, optimizer->datatype))
    {
        error = tensor_constant(optimizer->weight_decay, datatype, runtime, false, false, &weight_decay);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(weight_decay, parameters, &weight_decay_product);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(weight_decay_product, parameters->gradient, &weight_decay_sum);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }

        tensor_destroy(parameters->gradient);
        parameters->gradient = NULL;
        error = tensor_as_tensor(weight_decay_sum, &parameters->gradient);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = tensor_constant(optimizer->alpha, datatype, runtime, false, false, &alpha_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
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
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }
    
    error = tensor_multiplication(parameters->gradient, parameters->gradient, &squared_current_gradient);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(one_minus_alpha_constant, squared_current_gradient, &one_minus_alpha_product);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    if (map_contains(optimizer->square_average, key))
    {
        error = map_get(optimizer->square_average, key, (void **) &square_average_initial);
        if (error)
        {
            error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(alpha_constant, square_average_initial, &alpha_product);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(alpha_product, one_minus_alpha_product, &square_average);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }

        tensor_destroy(square_average_initial); 
        square_average_initial = NULL;
    }
    else
    {
        error = tensor_as_tensor(one_minus_alpha_product, &square_average);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed create tensor."), error);
            goto cleanup;
        }
    }

    error = map_set(optimizer->square_average, key, square_average);
    if (error)
    {
        error = ERROR(ERROR_SET, string_create("failed set map entry."), error);
        goto cleanup;
    }

    if (optimizer->centered)
    {
        error = tensor_multiplication(one_minus_alpha_constant, parameters->gradient, &centered_grad);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        if (map_contains(optimizer->average_gradient, key))
        {
            error = map_get(optimizer->average_gradient, key, (void **) &average_gradient_initial);
            if (error)
            {
                error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
                goto cleanup;
            }

            error = tensor_multiplication(average_gradient_initial, alpha_constant, &alpha_average_grad);
            if (error)
            {
                error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
                goto cleanup;
            }

            error = tensor_addition(alpha_average_grad, centered_grad, &updated_average_grad);
            if (error)
            {
                error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
                goto cleanup;
            }

            tensor_destroy(average_gradient_initial);
            average_gradient_initial = NULL;
        }
        else
        {
            error = tensor_as_tensor(centered_grad, &updated_average_grad);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }
        }

        error = map_set(optimizer->average_gradient, key, updated_average_grad);
        if (error)
        {
            error = ERROR(ERROR_SET, string_create("failed to set tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(updated_average_grad, updated_average_grad, &average_gradient_squared);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_subtraction(square_average, average_gradient_squared, &square_average_telda);
        if (error)
        {
            error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        error = tensor_as_tensor(square_average, &square_average_telda);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = tensor_square_root(square_average_telda, &square_average_telda_root);
    if (error)
    {
        error = ERROR(ERROR_SQUARE_ROOT, string_create("failed to perfrom square root on tensor."), error);
        goto cleanup;
    }
    
    error = tensor_constant(optimizer->epsilon, datatype, runtime, false, false, &epsilon_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(square_average_telda_root, epsilon_constant, &square_average_telda_epsilon);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_division(parameters->gradient, square_average_telda_epsilon, &temp_gradient);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    error = tensor_constant(optimizer->learning_rate, datatype, runtime, false, false, &learning_rate);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    if (!is_zero(optimizer->momentum, optimizer->datatype))
    {
        if (map_contains(optimizer->momentum_buffer, key))
        {
            error = tensor_constant(optimizer->momentum, datatype, runtime, false, false, &momentum_constant);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }

            error = map_get(optimizer->momentum_buffer, key, (void **) &initial_momentum);
            if (error)
            {
                error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
                goto cleanup;
            }

            error = tensor_multiplication(momentum_constant, initial_momentum, &momentum_const_buffer);
            if (error)
            {
                error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
                goto cleanup;
            }

            error = tensor_addition(momentum_const_buffer, temp_gradient, &updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
                goto cleanup;
            }

            tensor_destroy(initial_momentum);
            initial_momentum = NULL;

            error = map_set(optimizer->momentum_buffer, key, updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_SET, string_create("failed to set map entry."), error);
                goto cleanup;
            }
        }
        else
        {
            error = tensor_as_tensor(temp_gradient, &updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
                goto cleanup;
            }

            error = map_set(optimizer->momentum_buffer, key, updated_momentum);
            if (error)
            {
                error = ERROR(ERROR_SET, string_create("failed to set map entry."), error);
            }
        }

        error = tensor_multiplication(learning_rate, updated_momentum, &parameter_update);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        error = tensor_multiplication(learning_rate, temp_gradient, &parameter_update);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
    }

    error = tensor_subtraction(parameters, parameter_update, &parameters);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    with_no_gradient(false);

cleanup:
    string_destroy(key);
    tensor_destroy(learning_rate);
    tensor_destroy(weight_decay);
    tensor_destroy(weight_decay_product);
    tensor_destroy(alpha_constant);
    tensor_destroy(alpha_product);
    tensor_destroy(one_minus_alpha_constant);
    tensor_destroy(squared_current_gradient);
    tensor_destroy(square_average_initial);
    tensor_destroy(one_minus_alpha_product);
    tensor_destroy(square_average_telda);
    tensor_destroy(temp_optimizer_square_average);
    tensor_destroy(learning_rate_gradient);
    tensor_destroy(square_average_telda_root);
    tensor_destroy(epsilon_constant);
    tensor_destroy(square_average_telda_epsilon);
    tensor_destroy(parameter_update);
    tensor_destroy(momentum_const_buffer);
    tensor_destroy(temp_gradient);
    tensor_destroy(momentum_constant);
    tensor_destroy(momentum_product);
    tensor_destroy(modified_momentum);
    tensor_destroy(centered_grad);
    tensor_destroy(alpha_average_grad);
    tensor_destroy(average_gradient_squared);
    tensor_destroy(average_gradient);
    tensor_destroy(average_gradient_initial);
    tensor_destroy(weight_decay_sum);
    tensor_destroy(initial_momentum);
    return error;
}



nw_error_t *adam(adam_t *optimizer, tensor_t *parameters)
{
    CHECK_NULL_ARGUMENT(optimizer, "optimizer");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    nw_error_t *error = NULL;
    datatype_t datatype = parameters->buffer->storage->datatype;
    runtime_t runtime = parameters->buffer->storage->runtime;

    tensor_t *learning_rate = NULL;
    tensor_t *weight_decay = NULL;
    tensor_t *weight_decay_product = NULL;
    tensor_t *beta_1_constant = NULL;
    tensor_t *beta_2_constant = NULL;
    tensor_t *one_minus_beta_1_constant = NULL;
    tensor_t *one_minus_beta_2_constant = NULL;
    tensor_t *beta_1_constant_squared = NULL;
    tensor_t *beta_2_constant_squared = NULL;
    tensor_t *first_moment = NULL;
    tensor_t *first_moment_part_0 = NULL;
    tensor_t *first_moment_part_1 = NULL;
    tensor_t *first_moment_part_2 = NULL;
    tensor_t *gradient_squared = NULL;
    tensor_t *second_moment = NULL;
    tensor_t *second_moment_part_0 = NULL;
    tensor_t *second_moment_part_1 = NULL;
    tensor_t *second_moment_part_2 = NULL;
    tensor_t *first_momentum_telda = NULL;
    tensor_t *second_momentum_telda = NULL;
    tensor_t *epsilon_constant = NULL;
    tensor_t *square_root_max_moment = NULL;
    tensor_t *square_root_plus_epsilon = NULL;
    tensor_t *modified_learning_rate = NULL;
    tensor_t *parameter_update = NULL;
    tensor_t *temp_gradient_addition = NULL;
    int64_t *iteration = NULL;
    string_t key = string_create("%lu", parameters->id);

    if (map_contains(optimizer->iteration, key))
    {
        error = map_get(optimizer->iteration, key, (void **) &iteration);
        if (error)
        {
            error = ERROR(ERROR_GET, string_create("failed to get iteration."), error);
            goto cleanup;
        }
        (*iteration)++;
    }
    else
    {
        iteration = (int64_t *) malloc(sizeof(int64_t));
        if (!iteration)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(int64_t)), error);
            goto cleanup;
        }
        *iteration = 1;

        error = map_set(optimizer->iteration, key, (void *) iteration);
        if (error)
        {
            free(iteration);
            error = ERROR(ERROR_SET, string_create("failed to set map entry."), error);
            goto cleanup;
        }
    }

    error = tensor_constant(optimizer->learning_rate, datatype, runtime, false, false, &learning_rate);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    with_no_gradient(true);

    if (!is_zero(optimizer->weight_decay, optimizer->datatype))
    {
        error = tensor_constant(optimizer->weight_decay, datatype, runtime, false, false, &weight_decay);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(weight_decay, parameters, &weight_decay_product);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(weight_decay_product, parameters->gradient, &temp_gradient_addition);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }

        tensor_destroy(parameters->gradient);
        parameters->gradient = NULL;
        error = tensor_as_tensor(temp_gradient_addition, &parameters->gradient);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    switch(datatype)
    {
        case FLOAT32:
            float32_t beta_1_float32 = (float32_t) 1 - *(float32_t *) (optimizer->beta_1);
            float32_t beta_1_squared_float32 = (float32_t) 1 - powf(*(float32_t *) (optimizer->beta_1), *iteration);
            error = tensor_constant(&beta_1_float32, datatype, runtime, false, false, &one_minus_beta_1_constant);
            error = tensor_constant(optimizer->beta_1, datatype, runtime, false, false, &beta_1_constant);
            error = tensor_constant(&beta_1_squared_float32, datatype, runtime, false, false, &beta_1_constant_squared);

            float32_t beta_2_float32 = (float32_t) 1 - *(float32_t *) (optimizer->beta_2);
            float32_t beta_2_squared_float32 = (float32_t) 1 - powf(*(float32_t *) (optimizer->beta_2), *iteration);
            error = tensor_constant(&beta_2_float32, datatype, runtime, false, false, &one_minus_beta_2_constant);
            error = tensor_constant(optimizer->beta_2, datatype, runtime, false, false, &beta_2_constant);
            error = tensor_constant(&beta_2_squared_float32, datatype, runtime, false, false, &beta_2_constant_squared);
            break;
        case FLOAT64:
            float64_t beta_1_float64 = (float64_t) 1 - *(float64_t *) (optimizer->beta_1);
            float64_t beta_1_squared_float64 = (float64_t) 1 - pow(*(float64_t *) (optimizer->beta_1), *iteration);
            error = tensor_constant(&beta_1_float64, datatype, runtime, false, false, &one_minus_beta_1_constant);
            error = tensor_constant(optimizer->beta_1, datatype, runtime, false, false, &beta_1_constant);
            error = tensor_constant(&beta_1_squared_float64, datatype, runtime, false, false, &beta_1_constant_squared);

            float64_t beta_2_float64 = (float64_t) 1 - *(float64_t *) (optimizer->beta_2);
            float64_t beta_2_squared_float64 = (float64_t) 1 - pow(*(float64_t *) (optimizer->beta_2), *iteration);
            error = tensor_constant(&beta_2_float64, datatype, runtime, false, false, &one_minus_beta_2_constant);
            error = tensor_constant(optimizer->beta_2, datatype, runtime, false, false, &beta_2_constant);
            error = tensor_constant(&beta_2_squared_float64, datatype, runtime, false, false, &beta_2_constant_squared);
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
            goto cleanup;
    } 

    //first moment
    error = tensor_multiplication(one_minus_beta_1_constant, parameters->gradient, &first_moment_part_0);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    if (map_contains(optimizer->first_moment, key))
    {
        error = map_get(optimizer->first_moment, key, (void **) &first_moment_part_1);
        if (error)
        {
            error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(first_moment_part_1, beta_1_constant, &first_moment_part_2);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(first_moment_part_0, first_moment_part_2, &first_moment);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }

        tensor_destroy(first_moment_part_1);
        first_moment_part_1 = NULL;
    }
    else
    {
        error = tensor_as_tensor(first_moment_part_0, &first_moment);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = map_set(optimizer->first_moment, key, first_moment);
    if (error)
    {
        error = ERROR(ERROR_SET, string_create("failed to set map entry."), error);
        goto cleanup;
    }

    // second moment
    error = tensor_multiplication(parameters->gradient, parameters->gradient, &gradient_squared);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(one_minus_beta_2_constant, gradient_squared, &second_moment_part_0);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    if (map_contains(optimizer->second_moment, key))
    {
        error = map_get(optimizer->second_moment, key, (void **) &second_moment_part_1);
        if (error)
        {
            error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(second_moment_part_1, beta_2_constant, &second_moment_part_2);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(second_moment_part_0, second_moment_part_2, &second_moment);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }

        tensor_destroy(second_moment_part_1);
        second_moment_part_1 = NULL;
    }
    else
    {
        error = tensor_as_tensor(second_moment_part_0, &second_moment);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = map_set(optimizer->second_moment, key, second_moment);
    if (error)
    {
        error = ERROR(ERROR_SET, string_create("failed to set map entry."), error);
        goto cleanup;
    }

    //bias correction
    error = tensor_division(first_moment, beta_1_constant_squared, &first_momentum_telda);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    error = tensor_division(second_moment, beta_2_constant_squared, &second_momentum_telda);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    error = tensor_constant(optimizer->epsilon, datatype, runtime, false, false, &epsilon_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_square_root(second_momentum_telda, &square_root_max_moment);
    if (error)
    {
        error = ERROR(ERROR_SQUARE_ROOT, string_create("failed to perfrom square root on tensor."), error);
        goto cleanup;
    }
 
    error = tensor_addition(square_root_max_moment, epsilon_constant, &square_root_plus_epsilon);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(learning_rate, first_momentum_telda, &modified_learning_rate);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_division(modified_learning_rate, square_root_plus_epsilon, &parameter_update);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    error = tensor_subtraction(parameters, parameter_update, &parameters);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    } 

    with_no_gradient(false);

    error = NULL;
    goto cleanup;

cleanup:
    tensor_destroy(learning_rate);
    tensor_destroy(weight_decay);
    tensor_destroy(weight_decay_product);
    tensor_destroy(beta_1_constant);
    tensor_destroy(beta_2_constant);
    tensor_destroy(one_minus_beta_1_constant);
    tensor_destroy(one_minus_beta_2_constant);
    tensor_destroy(beta_1_constant_squared);
    tensor_destroy(beta_2_constant_squared);
    tensor_destroy(first_moment_part_0);
    tensor_destroy(first_moment_part_1);
    tensor_destroy(first_moment_part_2);
    tensor_destroy(gradient_squared);
    tensor_destroy(second_moment_part_0);
    tensor_destroy(second_moment_part_1);
    tensor_destroy(second_moment_part_2);
    tensor_destroy(first_momentum_telda);
    tensor_destroy(second_momentum_telda);
    tensor_destroy(epsilon_constant);
    tensor_destroy(square_root_max_moment);
    tensor_destroy(square_root_plus_epsilon);
    tensor_destroy(modified_learning_rate);
    tensor_destroy(parameter_update);
    tensor_destroy(temp_gradient_addition);
    string_destroy(key);
    return error;
    
}