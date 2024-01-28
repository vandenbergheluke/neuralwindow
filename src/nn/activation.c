#include <activation.h>

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
        return NULL;
    case ACTIVATION_SOFTMAX:
    case ACTIVATION_LOGSOFTMAX:
        (*activation_function)->softmax = (softmax_t *) type_activation_function;
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
    case ACTIVATION_SOFTMAX:
        return "ACTIVATION_SOFTMAX";
    case ACTIVATION_LOGSOFTMAX:
        return "ACTIVATION_LOGSOFTMAX";
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

nw_error_t *activation_forward(activation_t *activation, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_ACTIVATION("activation", activation);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

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
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown activation function %d.", (int) activation_function_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply activation function."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}
