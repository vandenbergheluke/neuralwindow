#include <activation.h>
#include <string.h>

nw_error_t *activation_create(activation_t **activation, activation_function_t *activation_function, activation_function_type_t activation_function_type)
{
    CHECK_NULL_ARGUMENT(activation, "activation");
    CHECK_NULL_ARGUMENT(activation_function, "activation_function");

    *activation = (activation_t *) malloc(sizeof(activation_t));
    if (!*activation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(activation_t)), NULL);
    }

    (*activation)->activation_function = activation_function;
    (*activation)->activation_function_type = activation_function_type;

    return NULL;
}

void activation_destroy(activation_t *activation)
{
    if (activation)
    {
        activation_function_destroy(activation->activation_function, activation->activation_function_type);
        free(activation);
    }
}

nw_error_t *activation_function_create(activation_function_t **activation_function,
                                       activation_function_type_t activation_function_type,
                                       void *type_activation_function)
{
    CHECK_NULL_ARGUMENT(activation_function, "activation_function");

    *activation_function = (activation_function_t *) malloc(sizeof(activation_function_t));
    if (!*activation_function)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(activation_function_t)), NULL);
    }

    switch (activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
    case ACTIVATION_SIGMOID:
    case ACTIVATION_TANH:
    case ACTIVATION_GELU:
        return NULL;
    case ACTIVATION_SOFTMAX:
    case ACTIVATION_LOGSOFTMAX:
        (*activation_function)->softmax = (softmax_t *) type_activation_function;
        break;
    case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
        (*activation_function)->leaky_rectified_linear = (leaky_rectified_linear_t *) type_activation_function;
        break;
    default:
        free(*activation_function);
        return ERROR(ERROR_ACTIVATION_TYPE, string_create("unknown activation type %d.", (int) activation_function_type), NULL);
    }

    return NULL;
}

void activation_function_destroy(activation_function_t *activation_function, activation_function_type_t activation_function_type)
{
    if (activation_function)
    {
        switch (activation_function_type)
        {
        case ACTIVATION_SOFTMAX:
        case ACTIVATION_LOGSOFTMAX:
            softmax_destroy(activation_function->softmax);
            break;
        case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
            leaky_rectified_linear_destroy(activation_function->leaky_rectified_linear);
            break;
        default:
            break;
        }
        free(activation_function);
    }
}

string_t activation_function_type_string(activation_function_type_t activation_function_type)
{
    switch (activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
        return "ACTIVATION_RECTIFIED_LINEAR";
    case ACTIVATION_SIGMOID:
        return "ACTIVATION_SIGMOID";
    case ACTIVATION_TANH:
        return "ACTIVATION_TANH";
    case ACTIVATION_GELU:
        return "ACTIVATION_GELU";
    case ACTIVATION_SOFTMAX:
        return "ACTIVATION_SOFTMAX";
    case ACTIVATION_LOGSOFTMAX:
        return "ACTIVATION_LOGSOFTMAX";
    case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
        return "ACTIVATION_LEAKY_RECTIFIED_LINEAR";
    default:
        return "ACTIVATION_FUNCTION_TYPE";
    }
}

nw_error_t *softmax_create(softmax_t **softmax, int64_t axis)
{
    CHECK_NULL_ARGUMENT(softmax, "softmax");

    *softmax = (softmax_t *) malloc(sizeof(softmax_t));
    if (!*softmax)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(softmax_t)), NULL);
    }

    (*softmax)->axis = axis;

    return NULL;
}

void softmax_destroy(softmax_t *softmax)
{
    if (softmax)
    {
        free(softmax);
    }
}

nw_error_t *leaky_rectified_linear_create(leaky_rectified_linear_t **leaky_rectified_linear, void *c, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(leaky_rectified_linear, "leaky_rectified_linear");
    CHECK_NULL_ARGUMENT(c, "c");

    *leaky_rectified_linear = (leaky_rectified_linear_t *) malloc(sizeof(leaky_rectified_linear_t));
    if (!*leaky_rectified_linear)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(leaky_rectified_linear_t)), NULL);
    }

    (*leaky_rectified_linear)->c = (void *) malloc(datatype_size(datatype));
    if (!*leaky_rectified_linear)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(datatype_size(datatype))), NULL);
    }

    memcpy((*leaky_rectified_linear)->c, c, datatype_size(datatype));
    (*leaky_rectified_linear)->datatype = datatype;

    return NULL;
}

void leaky_rectified_linear_destroy(leaky_rectified_linear_t *leaky_rectified_linear)
{
    if (leaky_rectified_linear)
    {
        free(leaky_rectified_linear->c);
        free(leaky_rectified_linear);
    }
}

nw_error_t *rectified_linear_activation_create(activation_t **activation)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_RECTIFIED_LINEAR;

    error = activation_function_create(&activation_function, activation_function_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *sigmoid_activation_create(activation_t **activation)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_SIGMOID;

    error = activation_function_create(&activation_function, activation_function_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *tanh_activation_create(activation_t **activation)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_TANH;

    error = activation_function_create(&activation_function, activation_function_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *gelu_activation_create(activation_t **activation)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_GELU;

    error = activation_function_create(&activation_function, activation_function_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

static nw_error_t *softmax_activation_type_create(activation_t **activation, int64_t axis, activation_function_type_t activation_function_type)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    softmax_t *softmax = NULL;
    activation_function_t *activation_function = NULL;

    error = softmax_create(&softmax, axis);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create softmax."), error);
    }

    error = activation_function_create(&activation_function, activation_function_type, softmax);
    if (error)
    {
        softmax_destroy(softmax);
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *softmax_activation_create(activation_t **activation, int64_t axis)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;

    error = softmax_activation_type_create(activation, axis, ACTIVATION_SOFTMAX);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create softmax activation."), error);
    }

    return error;
}

nw_error_t *logsoftmax_activation_create(activation_t **activation, int64_t axis)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;

    error = softmax_activation_type_create(activation, axis, ACTIVATION_LOGSOFTMAX);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create softmax activation."), error);
    }

    return error;
}

nw_error_t *leaky_rectified_linear_activation_create(activation_t **activation, void *c, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    leaky_rectified_linear_t *leaky_rectified_linear = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_LEAKY_RECTIFIED_LINEAR;

    error = leaky_rectified_linear_create(&leaky_rectified_linear, c, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create leaky rectified linear function."), error);
    }

    error = activation_function_create(&activation_function, activation_function_type, (void *) leaky_rectified_linear);
    if (error)
    {
        leaky_rectified_linear_destroy(leaky_rectified_linear);
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *activation_forward(activation_t *activation, tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(activation, "activation");
    CHECK_NULL_ARGUMENT(activation->activation_function, "activation->activation_function");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = activation->activation_function;
    activation_function_type_t activation_function_type = activation->activation_function_type;

    switch (activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
        error = tensor_rectified_linear(x, y);
        break;
    case ACTIVATION_SIGMOID:
        error = tensor_sigmoid(x, y);
        break;
    case ACTIVATION_TANH:
        error = tensor_tanh(x, y);
        break;
    case ACTIVATION_GELU:
        error = tensor_gelu(x, y);
        break;
    case ACTIVATION_SOFTMAX:
        if (!activation_function->softmax)
        {
            error = ERROR(ERROR_NULL, string_create("activation function is null."), NULL);
        }
        else
        {
            error = tensor_softmax(x, y, activation_function->softmax->axis);
        }
        break;
    case ACTIVATION_LOGSOFTMAX:
        if (!activation_function->softmax)
        {
            error = ERROR(ERROR_NULL, string_create("activation function is null."), NULL);
        }
        else
        {
            error = tensor_logsoftmax(x, y, activation_function->softmax->axis);
        }
        break;
    case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
        if (!activation_function->leaky_rectified_linear)
        {
            error = ERROR(ERROR_NULL, string_create("activation function is null."), NULL);
        }
        else
        {
            error = tensor_leaky_rectified_linear(x, activation_function->leaky_rectified_linear->c, y);
        }
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown activation function %d.", (int) activation_function_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply activation function."), error);
    }

    return error;
}
