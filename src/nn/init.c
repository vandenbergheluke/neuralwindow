#include <buffer.h>
#include <tensor.h>
#include <layer.h>
#include <init.h>
#include <math.h>
#include <string.h>

nw_error_t *parameter_init_create(parameter_init_t **parameter_init, init_t *init, init_type_t init_type)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");
    CHECK_NULL_ARGUMENT(init, "init");

    *parameter_init = (parameter_init_t *) malloc(sizeof(parameter_init_t));
    if (!*parameter_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(parameter_init_t)), NULL);
    }

    (*parameter_init)->init = init;
    (*parameter_init)->init_type = init_type;

    return NULL;
}

void parameter_init_destroy(parameter_init_t *parameter_init)
{
    if (parameter_init)
    {
        init_destroy(parameter_init->init, parameter_init->init_type);
        free(parameter_init);
    }
}

nw_error_t *init_create(init_t **init, init_type_t init_type, void *type_init)
{
    CHECK_NULL_ARGUMENT(init, "init");

    *init = (init_t *) malloc(sizeof(init_t));
    if (!*init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(init_t)), NULL);
    }

    switch (init_type)
    {
    case ZEROES:
    case ONES:
        return NULL;
    case UNIFORM:
        (*init)->uniform_init = (uniform_init_t *) type_init;
        break;
    case NORMAL:
        (*init)->normal_init = (normal_init_t *) type_init;
        break;
    case KAIMING_UNIFORM:
    case KAIMING_NORMAL:
        (*init)->kaiming_init = (kaiming_init_t *) type_init;
        break;
    case GLOROT_UNIFORM:
    case GLOROT_NORMAL:
        (*init)->glorot_init = (glorot_init_t *) type_init;
        break;
    default:
        free(*init);
        return ERROR(ERROR_LAYER_TYPE, string_create("unknown init type %d.", (int) init_type), NULL);
    }

    return NULL;
}

void init_destroy(init_t *init, init_type_t init_type)
{
    if (init)
    {
        switch (init_type)
        {
        case ZEROES:
        case ONES:
            break;
        case UNIFORM:
            uniform_init_destroy(init->uniform_init);
            break;
        case NORMAL:
            normal_init_destroy(init->normal_init);
            break;
        case KAIMING_UNIFORM:
        case KAIMING_NORMAL:
            kaiming_init_destroy(init->kaiming_init);
            break;
        case GLOROT_UNIFORM:
        case GLOROT_NORMAL:
            glorot_init_destroy(init->glorot_init);
            break;
        default:
            break;
        }
        free(init);
    }
}

nw_error_t *uniform_init_create(uniform_init_t **uniform_init, void *lower_bound, void *upper_bound, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(uniform_init, "uniform_init");
    CHECK_NULL_ARGUMENT(lower_bound, "lower_bound");
    CHECK_NULL_ARGUMENT(upper_bound, "upper_bound");

    *uniform_init = (uniform_init_t *) malloc(sizeof(uniform_init_t));
    if (!*uniform_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(uniform_init_t)), NULL);
    }

    size_t size = datatype_size(datatype);
    nw_error_t *error = NULL;
    (*uniform_init)->lower_bound = NULL;
    (*uniform_init)->upper_bound = NULL;

    (*uniform_init)->upper_bound = (void *) malloc(size);
    if (!(*uniform_init)->upper_bound)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*uniform_init)->upper_bound, upper_bound, size);

    (*uniform_init)->lower_bound = (void *) malloc(size);
    if (!(*uniform_init)->lower_bound)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*uniform_init)->lower_bound, lower_bound, size);

    return error;

cleanup:

    uniform_init_destroy(*uniform_init);

    return error;
}

void uniform_init_destroy(uniform_init_t *uniform_init)
{
    if (uniform_init)
    {
        free(uniform_init->lower_bound);
        free(uniform_init->upper_bound);
        free(uniform_init);
    }
}

nw_error_t *normal_init_create(normal_init_t **normal_init, void *mean, void *standard_deviation, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(normal_init, "normal_init");
    CHECK_NULL_ARGUMENT(mean, "mean");
    CHECK_NULL_ARGUMENT(standard_deviation, "standard_deviation");

    *normal_init = (normal_init_t *) malloc(sizeof(normal_init_t));
    if (!*normal_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(normal_init_t)), NULL);
    }

    size_t size = datatype_size(datatype);
    nw_error_t *error = NULL;
    (*normal_init)->mean = NULL;
    (*normal_init)->standard_deviation = NULL;

    (*normal_init)->standard_deviation = (void *) malloc(size);
    if (!(*normal_init)->standard_deviation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*normal_init)->standard_deviation, standard_deviation, size);

    (*normal_init)->mean = (void *) malloc(size);
    if (!(*normal_init)->mean)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*normal_init)->mean, mean, size);

    return error;

cleanup:

    normal_init_destroy(*normal_init);

    return error;
}

void normal_init_destroy(normal_init_t *normal_init)
{
    if (normal_init)
    {
        free(normal_init->mean);
        free(normal_init->standard_deviation);
        free(normal_init);
    }
}

nw_error_t *kaiming_init_create(kaiming_init_t **kaiming_init, void *gain, bool_t mode, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(kaiming_init, "kaiming_init");
    CHECK_NULL_ARGUMENT(gain, "gain");

    *kaiming_init = (kaiming_init_t *) malloc(sizeof(kaiming_init_t));
    if (!*kaiming_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(kaiming_init_t)), NULL);
    }

    size_t size = datatype_size(datatype);
    nw_error_t *error = NULL;
    (*kaiming_init)->mode = mode;
    (*kaiming_init)->gain = NULL;

    (*kaiming_init)->gain = (void *) malloc(size);
    if (!(*kaiming_init)->gain)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*kaiming_init)->gain, gain, size);

    return error;

cleanup:

    kaiming_init_destroy(*kaiming_init);

    return error;
}

void kaiming_init_destroy(kaiming_init_t *kaiming_init)
{
    if (kaiming_init)
    {
        free(kaiming_init->gain);
        free(kaiming_init);
    }
}

nw_error_t *glorot_init_create(glorot_init_t **glorot_init, void *gain, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(glorot_init, "glorot_init");
    CHECK_NULL_ARGUMENT(gain, "gain");

    *glorot_init = (glorot_init_t *) malloc(sizeof(glorot_init_t));
    if (!*glorot_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(glorot_init_t)), NULL);
    }

    size_t size = datatype_size(datatype);
    nw_error_t *error = NULL;
    (*glorot_init)->gain = NULL;

    (*glorot_init)->gain = (void *) malloc(size);
    if (!(*glorot_init)->gain)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*glorot_init)->gain, gain, size);

    return error;

cleanup:

    glorot_init_destroy(*glorot_init);

    return error;
}

void glorot_init_destroy(glorot_init_t *glorot_init)
{
    if (glorot_init)
    {
        free(glorot_init->gain);
        free(glorot_init);
    }
}

nw_error_t *zeroes_parameter_init(parameter_init_t **parameter_init)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    init_t *init = NULL;
    init_type_t init_type = ZEROES;

    error = init_create(&init, init_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *ones_parameter_init(parameter_init_t **parameter_init)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    init_t *init = NULL;
    init_type_t init_type = ONES;

    error = init_create(&init, init_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *uniform_parameter_init(parameter_init_t **parameter_init, void *lower_bound, void *upper_bound, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    uniform_init_t *uniform_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = UNIFORM;

    error = uniform_init_create(&uniform_init, lower_bound, upper_bound, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create uniform initializer."), error);
    }

    error = init_create(&init, init_type, (void *) uniform_init);
    if (error)
    {
        uniform_init_destroy(uniform_init);
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *normal_parameter_init(parameter_init_t **parameter_init, void *mean, void *standard_deviation, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    normal_init_t *normal_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = NORMAL;

    error = normal_init_create(&normal_init, mean, standard_deviation, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create normal initializer."), error);
    }

    error = init_create(&init, init_type, (void *) normal_init);
    if (error)
    {
        normal_init_destroy(normal_init);
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *kaiming_uniform_parameter_init(parameter_init_t **parameter_init, void *gain, bool_t mode, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    kaiming_init_t *kaiming_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = KAIMING_UNIFORM;

    error = kaiming_init_create(&kaiming_init, gain, mode, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create kaiming initializer."), error);
    }

    error = init_create(&init, init_type, (void *) kaiming_init);
    if (error)
    {
        kaiming_init_destroy(kaiming_init);
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *kaiming_normal_parameter_init(parameter_init_t **parameter_init, void *gain, bool_t mode, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    kaiming_init_t *kaiming_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = KAIMING_NORMAL;

    error = kaiming_init_create(&kaiming_init, gain, mode, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create kaiming initializer."), error);
    }

    error = init_create(&init, init_type, (void *) kaiming_init);
    if (error)
    {
        kaiming_init_destroy(kaiming_init);
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *glorot_uniform_parameter_init(parameter_init_t **parameter_init, void *gain, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    glorot_init_t *glorot_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = GLOROT_UNIFORM;

    error = glorot_init_create(&glorot_init, gain, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create glorot initializer."), error);
    }

    error = init_create(&init, init_type, (void *) glorot_init);
    if (error)
    {
        glorot_init_destroy(glorot_init);
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *glorot_normal_parameter_init(parameter_init_t **parameter_init, void *gain, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    glorot_init_t *glorot_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = GLOROT_NORMAL;

    error = glorot_init_create(&glorot_init, gain, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create glorot initializer."), error);
    }

    error = init_create(&init, init_type, (void *) glorot_init);
    if (error)
    {
        glorot_init_destroy(glorot_init);
        return ERROR(ERROR_CREATE, string_create("failed to create initializer."), error);
    }

    error = parameter_init_create(parameter_init, init, init_type);
    if (error)
    {
        init_destroy(init, init_type);
        return ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    }

    return error;
}

nw_error_t *initialize(tensor_t **parameters,
                       parameter_init_t *parameter_init,
                       const int64_t *shape,
                       int64_t rank,
                       runtime_t runtime,
                       datatype_t datatype,
                       bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");
    CHECK_NULL_ARGUMENT(parameter_init->init, "parameter_init->init");

    nw_error_t *error = NULL;
    init_type_t init_type = parameter_init->init_type;
    init_t *init = parameter_init->init;
    bool_t persist = true;

    switch (init_type)
    {
    case ZEROES:
        error = tensor_create_zeroes(parameters, shape, rank, runtime, datatype, requires_gradient, persist);
        break;
    case ONES:
        error = tensor_create_ones(parameters, shape, rank, runtime, datatype, requires_gradient, persist);
        break;
    case UNIFORM:
        if (init->uniform_init)
        {
            void *lower_bound = init->uniform_init->lower_bound;
            void *upper_bound = init->uniform_init->upper_bound;
            error = tensor_create_uniform(parameters, shape, rank, runtime, datatype, requires_gradient, persist, lower_bound, upper_bound);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case NORMAL:
        if (init->normal_init)
        {
            void *mean = init->normal_init->mean;
            void *standard_deviation = init->normal_init->standard_deviation;
            error = tensor_create_normal(parameters, shape, rank, runtime, datatype, requires_gradient, persist, mean, standard_deviation);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case KAIMING_UNIFORM:
        if (init->kaiming_init)
        {
            bool_t mode = init->kaiming_init->mode;
            void *gain = init->kaiming_init->gain;
            error = tensor_create_kaiming_uniform(parameters, shape, rank, runtime, datatype, requires_gradient, persist, gain, mode);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case KAIMING_NORMAL:
        if (init->kaiming_init)
        {
            bool_t mode = init->kaiming_init->mode;
            void *gain = init->kaiming_init->gain;
            error = tensor_create_kaiming_normal(parameters, shape, rank, runtime, datatype, requires_gradient, persist, gain, mode);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case GLOROT_UNIFORM:
        if (init->glorot_init)
        {
            void *gain = init->glorot_init->gain;
            error = tensor_create_glorot_uniform(parameters, shape, rank, runtime, datatype, requires_gradient, persist, gain);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case GLOROT_NORMAL:
        if (init->normal_init)
        {
            void *gain = init->glorot_init->gain;
            error = tensor_create_glorot_normal(parameters, shape, rank, runtime, datatype, requires_gradient, persist, gain);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    default:
        error = ERROR(ERROR_INITIALIZATION, string_create("unknown init type %d.", (int) init_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize parameters."), error);
    }

    return error;
}

nw_error_t *calculate_gain(activation_function_type_t activation_function_type, datatype_t datatype, void *gain, void *c)
{
    CHECK_NULL_ARGUMENT(gain, "gain");

    switch (activation_function_type)
    {
    case ACTIVATION_SIGMOID:
    case ACTIVATION_SOFTMAX:
        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) gain = (float32_t) 1.0;
            break;
        case FLOAT64:
            *(float64_t *) gain = (float64_t) 1.0;
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), NULL);
        }
        break;
    case ACTIVATION_RECTIFIED_LINEAR:
        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) gain = sqrtf((float32_t) 2.0);
            break;
        case FLOAT64:
            *(float64_t *) gain = sqrt((float64_t) 2.0);
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), NULL);
        }
        break;
    case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) gain = sqrtf((float32_t) 2.0 / ((float32_t) 1.0 + powf(*(float32_t *) c, (float32_t) 2.0)));
            break;
        case FLOAT64:
            *(float64_t *) gain = sqrt((float64_t) 2.0 / ((float64_t) 1.0 + pow(*(float64_t *) c, (float64_t) 2.0)));
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), NULL);
        }
        break;
    case ACTIVATION_TANH:
        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) gain = (float32_t) 5.0 / (float32_t) 3.0;
            break;
        case FLOAT64:
            *(float64_t *) gain = (float64_t) 5.0 / (float64_t) 3.0;
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), NULL);
        }
        break;
    default:
        return ERROR(ERROR_ACTIVATION_TYPE, string_create("unsupported activation type %d.", (int) activation_function_type), NULL);
    }

    return NULL;
}