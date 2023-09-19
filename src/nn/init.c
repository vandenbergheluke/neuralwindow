#include <buffer.h>
#include <tensor.h>
#include <layer.h>
#include <init.h>

nw_error_t *parameter_initialization_create(parameter_initialization_t **parameter_initialization, initialization_t *initialization, initialization_type_t initialization_type)
{
    CHECK_NULL_ARGUMENT(parameter_initialization, "parameter_initialization");
    CHECK_NULL_ARGUMENT(initialization, "initialization");

    *parameter_initialization = (parameter_initialization_t *) malloc(sizeof(parameter_initialization_t));
    if (!*parameter_initialization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(parameter_initialization_t)), NULL);
    }

    (*parameter_initialization)->initialization = initialization;
    (*parameter_initialization)->initialization_type = initialization_type;

    return NULL;
}

void parameter_initialization_destroy(parameter_initialization_t *parameter_initialization)
{
    if (parameter_initialization)
    {
        initialization_destroy(parameter_initialization->initialization, parameter_initialization->initialization_type);
        free(parameter_initialization);
    }
}

nw_error_t *initialization_create(initialization_t **initialization, initialization_type_t initialization_type, void *type_initialization)
{
    CHECK_NULL_ARGUMENT(initialization, "initialization");
    CHECK_NULL_ARGUMENT(type_initialization, "type_initialization");

    *initialization = (initialization_t *) malloc(sizeof(initialization_t));
    if (!*initialization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(initialization_t)), NULL);
    }

    switch (initialization_type)
    {
    case ZEROES:
    case ONES:
        return NULL;
    case UNIFORM:
        (*initialization)->uniform_initialization = (uniform_initialization_t *) type_initialization;
        break;
    case NORMAL:
        (*initialization)->normal_initialization = (normal_initialization_t *) type_initialization;
        break;
    case KAIMING_UNIFORM:
    case KAIMING_NORMAL:
        (*initialization)->kaiming_initialization = (kaiming_initialization_t *) type_initialization;
        break;
    case GLOROT_UNIFORM:
    case GLOROT_NORMAL:
        (*initialization)->glorot_initialization = (glorot_initialization_t *) type_initialization;
        break;
    default:
        free(*initialization);
        return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown initialization type %d.", (int) initialization_type), NULL);
    }

    return NULL;
}

void initialization_destroy(initialization_t *initialization, initialization_type_t initialization_type)
{
    if (initialization)
    {
        switch (initialization_type)
        {
        case ZEROES:
        case ONES:
            return NULL;
        case UNIFORM:
            uniform_initialization_destroy(initialization->uniform_initialization);
            break;
        case NORMAL:
            normal_initialization_destroy(initialization->normal_initialization);
            break;
        case KAIMING_UNIFORM:
        case KAIMING_NORMAL:
            kaiming_initialization_destroy(initialization->kaiming_initialization);
            break;
        case GLOROT_UNIFORM:
        case GLOROT_NORMAL:
            glorot_initialization_destroy(initialization->glorot_initialization);
            break;
        default:
            break;
        }
        free(initialization);
    }
}

nw_error_t *uniform_initialization_create(uniform_initialization_t **uniform_initialization, void *lower_bound, void *upper_bound)
{
    CHECK_NULL_ARGUMENT(uniform_initialization, "uniform_initialization");
    CHECK_NULL_ARGUMENT(lower_bound, "lower_bound");
    CHECK_NULL_ARGUMENT(upper_bound, "upper_bound");

    *uniform_initialization = (uniform_initialization_t *) malloc(sizeof(uniform_initialization_t));
    if (!*uniform_initialization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes."), sizeof(uniform_initialization_t));
    }

    (*uniform_initialization)->lower_bound = lower_bound;
    (*uniform_initialization)->upper_bound = upper_bound;

    return NULL;
}

void uniform_initialization_destroy(uniform_initialization_t *uniform_initialization)
{
    if (uniform_initialization)
    {
        free(uniform_initialization);
    }
}

nw_error_t *normal_initialization_create(normal_initialization_t **normal_initialization, void *mean, void *standard_deviation)
{
    CHECK_NULL_ARGUMENT(normal_initialization, "normal_initialization");
    CHECK_NULL_ARGUMENT(mean, "mean");
    CHECK_NULL_ARGUMENT(standard_deviation, "standard_deviation");

    *normal_initialization = (normal_initialization_t *) malloc(sizeof(normal_initialization_t));
    if (!*normal_initialization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes."), sizeof(normal_initialization_t));
    }

    (*normal_initialization)->mean = mean;
    (*normal_initialization)->standard_deviation = standard_deviation;

    return NULL;
}

void normal_initialization_destroy(normal_initialization_t *normal_initialization)
{
    if (normal_initialization)
    {
        free(normal_initialization);
    }
}

nw_error_t *kaiming_initialization_create(kaiming_initialization_t **kaiming_initialization, void *fan, void *gain)
{
    CHECK_NULL_ARGUMENT(kaiming_initialization, "kaiming_initialization");
    CHECK_NULL_ARGUMENT(fan, "fan");
    CHECK_NULL_ARGUMENT(gain, "gain");

    *kaiming_initialization = (kaiming_initialization_t *) malloc(sizeof(kaiming_initialization_t));
    if (!*kaiming_initialization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes."), sizeof(kaiming_initialization_t));
    }

    (*kaiming_initialization)->fan = fan;
    (*kaiming_initialization)->gain = gain;

    return NULL;
}

void kaiming_initialization_destroy(kaiming_initialization_t *kaiming_initialization)
{
    if (kaiming_initialization)
    {
        free(kaiming_initialization);
    }
}

nw_error_t *glorot_initialization_create(glorot_initialization_t **glorot_initialization, void *fan_in, void *fan_out, void *gain)
{
    CHECK_NULL_ARGUMENT(glorot_initialization, "glorot_initialization");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");
    CHECK_NULL_ARGUMENT(fan_out, "fan_out");
    CHECK_NULL_ARGUMENT(gain, "gain");

    *glorot_initialization = (glorot_initialization_t *) malloc(sizeof(glorot_initialization_t));
    if (!*glorot_initialization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes."), sizeof(glorot_initialization_t));
    }

    (*glorot_initialization)->fan_in = fan_in;
    (*glorot_initialization)->fan_out = fan_out;
    (*glorot_initialization)->gain = gain;

    return NULL;
}

void glorot_initialization_destroy(glorot_initialization_t *glorot_initialization)
{
    if (glorot_initialization)
    {
        free(glorot_initialization);
    }
}

nw_error_t *initialize(tensor_t **parameters,
                       parameter_initialization_t *parameter_initialization,
                       const uint64_t *shape,
                       uint64_t rank,
                       runtime_t runtime,
                       datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(parameter_initialization, "parameter_initialization");
    CHECK_NULL_ARGUMENT(parameter_initialization->initialization, "parameter_initialization->initialization");

    nw_error_t *error = NULL;
    initialization_type_t initialization_type = parameter_initialization->initialization_type;
    initialization_t *initialization = parameter_initialization->initialization;

    switch (initialization_type)
    {
    case ZEROES:
        error = tensor_create_zeroes(parameters, shape, rank, runtime, datatype, true);
        break;
    case ONES:
        error = tensor_create_ones(parameters, shape, rank, runtime, datatype, true);
        break;
    case UNIFORM:
        if (initialization->uniform_initialization)
        {
            void *lower_bound = initialization->uniform_initialization->lower_bound;
            void *upper_bound = initialization->uniform_initialization->upper_bound;
            error = tensor_create_uniform(parameters, shape, rank, runtime, datatype, true, lower_bound, upper_bound);
        }
        else
        {
            return ERROR(ERROR_NULL, string_create("parameter initialization is null."), NULL);
        }
        break;
    case NORMAL:
        if (initialization->normal_initialization)
        {
            void *mean = initialization->normal_initialization->mean;
            void *standard_deviation = initialization->normal_initialization->standard_deviation;
            error = tensor_create_normal(parameters, shape, rank, runtime, datatype, true, mean, standard_deviation);
        }
        else
        {
            return ERROR(ERROR_NULL, string_create("parameter initialization is null."), NULL);
        }
        break;
    case KAIMING_UNIFORM:
        if (initialization->kaiming_initialization)
        {
            void *fan = initialization->kaiming_initialization->fan;
            void *gain = initialization->kaiming_initialization->gain;
            error = tensor_create_kaiming_uniform(parameters, shape, rank, runtime, datatype, true, lower_bound, upper_bound);
        }
        else
        {
            return ERROR(ERROR_NULL, string_create("parameter initialization is null."), NULL);
        }
        break;
    case KAIMING_NORMAL:
        if (initialization->normal_initialization)
        {
            void *gain;
            gain_type_t gain_type = initialization->kaiming_initialization->gain_type;
            void *a = initialization->kaiming_initialization->a;
            error = calculate_gain()
            void *standard_deviation = initialization->normal_initialization->standard_deviation;
            error = tensor_create_uniform(parameters, shape, rank, runtime, datatype, true, mean, standard_deviation);
        }
        else
        {
            return ERROR(ERROR_NULL, string_create("parameter initialization is null."), NULL);
        }
        break;
    case GLOROT_UNIFORM:
        if (initialization->uniform_initialization)
        {
            void *lower_bound = initialization->uniform_initialization->lower_bound;
            void *upper_bound = initialization->uniform_initialization->upper_bound;
            error = tensor_create_uniform(parameters, shape, rank, runtime, datatype, true, lower_bound, upper_bound);
        }
        else
        {
            return ERROR(ERROR_NULL, string_create("parameter initialization is null."), NULL);
        }
        break;
    case GLOROT_NORMAL:
        if (initialization->normal_initialization)
        {
            void *mean = initialization->normal_initialization->mean;
            void *standard_deviation = initialization->normal_initialization->standard_deviation;
            error = tensor_create_uniform(parameters, shape, rank, runtime, datatype, true, mean, standard_deviation);
        }
        else
        {
            return ERROR(ERROR_NULL, string_create("parameter initialization is null."), NULL);
        }
        break;
    default:
        break;
    }
}

nw_error_t *calculate_gain(activation_t activation, datatype_t datatype, void *a, void *gain)
{
    CHECK_NULL_ARGUMENT(gain, "gain");

    switch (activation)
    {
    case ACTIVATION_SIGMOID:
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
    // case HYPERBOLIC_TANGENT_GAIN:
    //     switch (datatype)
    //     {
    //     case FLOAT32:
    //         *(float32_t *) gain = (float32_t) 5.0 / (float32_t) 3.0;
    //         break;
    //     case FLOAT64:
    //         *(float64_t *) gain = (float64_t) 5.0 / (float64_t) 3.0;
    //         break;
    //     default:
    //         return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), NULL);
    //     }
    //     break;
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
    // case LEAKY_RECTIFIED_LINEAR_GAIN:
    //     switch (datatype)
    //     {
    //     case FLOAT32:
    //         *(float32_t *) gain = sqrtf((float32_t) 2.0 / ((float32_t) 1.0 + (*(float32_t *) a) * (*(float32_t *) a)));
    //         break;
    //     case FLOAT64:
    //         *(float64_t *) gain = sqrt((float64_t) 2.0 / ((float64_t) 1.0 + (*(float64_t *) a) * (*(float64_t *) a)));
    //         break;
    //     default:
    //         return ERROR(ERROR_DATATYPE, string_create("unsupported datatype %d.", (int) datatype), NULL);
    //     }
    //     break;
    default:
        return ERROR(ERROR_GAIN, string_create("unsupported datatype %d.", (int) datatype), NULL);
    }

    return NULL;
}