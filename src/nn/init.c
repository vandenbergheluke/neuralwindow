#include <buffer.h>
#include <tensor.h>
#include <layer.h>
#include <init.h>
#include <math.h>

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
        return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown init type %d.", (int) init_type), NULL);
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

nw_error_t *uniform_init_create(uniform_init_t **uniform_init, void *lower_bound, void *upper_bound)
{
    CHECK_NULL_ARGUMENT(uniform_init, "uniform_init");
    CHECK_NULL_ARGUMENT(lower_bound, "lower_bound");
    CHECK_NULL_ARGUMENT(upper_bound, "upper_bound");

    *uniform_init = (uniform_init_t *) malloc(sizeof(uniform_init_t));
    if (!*uniform_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(uniform_init_t)), NULL);
    }

    (*uniform_init)->lower_bound = lower_bound;
    (*uniform_init)->upper_bound = upper_bound;

    return NULL;
}

void uniform_init_destroy(uniform_init_t *uniform_init)
{
    if (uniform_init)
    {
        free(uniform_init);
    }
}

nw_error_t *normal_init_create(normal_init_t **normal_init, void *mean, void *standard_deviation)
{
    CHECK_NULL_ARGUMENT(normal_init, "normal_init");
    CHECK_NULL_ARGUMENT(mean, "mean");
    CHECK_NULL_ARGUMENT(standard_deviation, "standard_deviation");

    *normal_init = (normal_init_t *) malloc(sizeof(normal_init_t));
    if (!*normal_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(normal_init_t)), NULL);
    }

    (*normal_init)->mean = mean;
    (*normal_init)->standard_deviation = standard_deviation;

    return NULL;
}

void normal_init_destroy(normal_init_t *normal_init)
{
    if (normal_init)
    {
        free(normal_init);
    }
}

nw_error_t *kaiming_init_create(kaiming_init_t **kaiming_init, void *fan, void *gain)
{
    CHECK_NULL_ARGUMENT(kaiming_init, "kaiming_init");
    CHECK_NULL_ARGUMENT(fan, "fan");
    CHECK_NULL_ARGUMENT(gain, "gain");

    *kaiming_init = (kaiming_init_t *) malloc(sizeof(kaiming_init_t));
    if (!*kaiming_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(kaiming_init_t)), NULL);
    }

    (*kaiming_init)->fan = fan;
    (*kaiming_init)->gain = gain;

    return NULL;
}

void kaiming_init_destroy(kaiming_init_t *kaiming_init)
{
    if (kaiming_init)
    {
        free(kaiming_init);
    }
}

nw_error_t *glorot_init_create(glorot_init_t **glorot_init, void *fan_in, void *fan_out, void *gain)
{
    CHECK_NULL_ARGUMENT(glorot_init, "glorot_init");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");
    CHECK_NULL_ARGUMENT(fan_out, "fan_out");
    CHECK_NULL_ARGUMENT(gain, "gain");

    *glorot_init = (glorot_init_t *) malloc(sizeof(glorot_init_t));
    if (!*glorot_init)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(glorot_init_t)), NULL);
    }

    (*glorot_init)->fan_in = fan_in;
    (*glorot_init)->fan_out = fan_out;
    (*glorot_init)->gain = gain;

    return NULL;
}

void glorot_init_destroy(glorot_init_t *glorot_init)
{
    if (glorot_init)
    {
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

nw_error_t *uniform_parameter_init(parameter_init_t **parameter_init, void *lower_bound, void *upper_bound)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    uniform_init_t *uniform_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = UNIFORM;

    error = uniform_init_create(&uniform_init, lower_bound, upper_bound);
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

nw_error_t *normal_parameter_init(parameter_init_t **parameter_init, void *mean, void *standard_deviation)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    normal_init_t *normal_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = NORMAL;

    error = normal_init_create(&normal_init, mean, standard_deviation);
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

nw_error_t *kaiming_uniform_parameter_init(parameter_init_t **parameter_init, void *fan, void *gain)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    kaiming_init_t *kaiming_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = KAIMING_UNIFORM;

    error = kaiming_init_create(&kaiming_init, fan, gain);
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

nw_error_t *kaiming_normal_parameter_init(parameter_init_t **parameter_init, void *fan, void *gain)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    kaiming_init_t *kaiming_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = KAIMING_NORMAL;

    error = kaiming_init_create(&kaiming_init, fan, gain);
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

nw_error_t *glorot_uniform_parameter_init(parameter_init_t **parameter_init, void *fan_in, void *fan_out, void *gain)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    glorot_init_t *glorot_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = GLOROT_UNIFORM;

    error = glorot_init_create(&glorot_init, fan_in, fan_out, gain);
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

nw_error_t *glorot_normal_parameter_init(parameter_init_t **parameter_init, void *fan_in, void *fan_out, void *gain)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");

    nw_error_t *error = NULL;
    glorot_init_t *glorot_init = NULL;
    init_t *init = NULL;
    init_type_t init_type = GLOROT_NORMAL;

    error = glorot_init_create(&glorot_init, fan_in, fan_out, gain);
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
                       const uint64_t *shape,
                       uint64_t rank,
                       runtime_t runtime,
                       datatype_t datatype,
                       bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(parameter_init, "parameter_init");
    CHECK_NULL_ARGUMENT(parameter_init->init, "parameter_init->init");

    nw_error_t *error = NULL;
    init_type_t init_type = parameter_init->init_type;
    init_t *init = parameter_init->init;

    switch (init_type)
    {
    case ZEROES:
        error = tensor_create_zeroes(parameters, shape, rank, runtime, datatype, requires_gradient);
        break;
    case ONES:
        error = tensor_create_ones(parameters, shape, rank, runtime, datatype, requires_gradient);
        break;
    case UNIFORM:
        if (init->uniform_init)
        {
            void *lower_bound = init->uniform_init->lower_bound;
            void *upper_bound = init->uniform_init->upper_bound;
            error = tensor_create_uniform(parameters, shape, rank, runtime, datatype, requires_gradient, lower_bound, upper_bound);
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
            error = tensor_create_normal(parameters, shape, rank, runtime, datatype, requires_gradient, mean, standard_deviation);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case KAIMING_UNIFORM:
        if (init->kaiming_init)
        {
            void *fan = init->kaiming_init->fan;
            void *gain = init->kaiming_init->gain;
            error = tensor_create_kaiming_uniform(parameters, shape, rank, runtime, datatype, requires_gradient, gain, fan);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case KAIMING_NORMAL:
        if (init->kaiming_init)
        {
            void *fan = init->kaiming_init->fan;
            void *gain = init->kaiming_init->gain;
            error = tensor_create_kaiming_normal(parameters, shape, rank, runtime, datatype, requires_gradient, gain, fan);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case GLOROT_UNIFORM:
        if (init->glorot_init)
        {
            void *fan_in = init->glorot_init->fan_in;
            void *fan_out = init->glorot_init->fan_out;
            void *gain = init->glorot_init->gain;
            error = tensor_create_glorot_uniform(parameters, shape, rank, runtime, datatype, requires_gradient, gain, fan_in, fan_out);
        }
        else
        {
            error = ERROR(ERROR_NULL, string_create("parameter init is null."), NULL);
        }
        break;
    case GLOROT_NORMAL:
        if (init->normal_init)
        {
            void *fan_in = init->glorot_init->fan_in;
            void *fan_out = init->glorot_init->fan_out;
            void *gain = init->glorot_init->gain;
            error = tensor_create_glorot_normal(parameters, shape, rank, runtime, datatype, requires_gradient, gain, fan_in, fan_out);
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

nw_error_t *calculate_gain(activation_function_type_t activation_function_type, datatype_t datatype, void *gain)
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
            *(float32_t *) gain = sqrtf(2.0);
            break;
        case FLOAT64:
            *(float64_t *) gain = sqrt(2.0);
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