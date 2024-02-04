#include <operation.h>

/**
 * @brief Get string representation of `operation_type`.
 * @param operation_type Operation type to display as string.
 * @return A string literal representing the `operation_type`.
 */
string_t operation_type_string(operation_type_t operation_type)
{
    switch (operation_type)
    {
    case UNARY_OPERATION:
        return "UNARY_OPERATION";
    case BINARY_OPERATION:
        return "BINARY_OPERATION";
    case TERNARY_OPERATION:
        return "TERNARY_OPERATION";
    case REDUCTION_OPERATION:
        return "REDUCTION_OPERATION";
    case STRUCTURE_OPERATION:
        return "STRUCTURE_OPERATION";
    case CREATION_OPERATION:
        return "CREATION_OPERATION";
    default:
        return "OPERATION";
    }
}

/**
 * @brief Get string representation of `unary_operation_type`.
 * @param operation_type Operation type to display as string.
 * @return A string literal representing the `unary_operation_type`.
 */
string_t unary_operation_type_string(unary_operation_type_t unary_operation_type)
{
    switch (unary_operation_type)
    {
    case EXPONENTIAL_OPERATION:
        return "EXPONENTIAL_OPERATION";
    case LOGARITHM_OPERATION:
        return "LOGARITHM_OPERATION";
    case SINE_OPERATION:
        return "SINE_OPERATION";
    case COSINE_OPERATION:
        return "COSINE_OPERATION";
    case SQUARE_ROOT_OPERATION:
        return "SQUARE_ROOT_OPERATION";
    case RECIPROCAL_OPERATION:
        return "RECIPROCAL_OPERATION";
    case CONTIGUOUS_OPERATION:
        return "CONTIGUOUS_OPERATION";
    case NEGATION_OPERATION:
        return "NEGATION_OPERATION";
    case RECTIFIED_LINEAR_OPERATION:
        return "RECTIFIED_LINEAR_OPERATION";
    case SIGMOID_OPERATION:
        return "SIGMOID_OPERATION";
    case AS_OPERATION:
        return "AS_OPERATION";
    default:
        return "OPERATION";
    }
}

string_t binary_operation_type_string(binary_operation_type_t binary_operation_type)
{
    switch (binary_operation_type)
    {
    case ADDITION_OPERATION:
        return "ADDITION_OPERATION";
    case SUBTRACTION_OPERATION:
        return "SUBTRACTION_OPERATION";
    case MULTIPLICATION_OPERATION:
        return "MULTIPLICATION_OPERATION";
    case DIVISION_OPERATION:
        return "DIVISION_OPERATION";
    case POWER_OPERATION:
        return "POWER_OPERATION";
    case MATRIX_MULTIPLICATION_OPERATION:
        return "MATRIX_MULTIPLICATION_OPERATION";
    case COMPARE_EQUAL_OPERATION:
        return "COMPARE_EQUAL_OPERATION";
    case COMPARE_GREATER_OPERATION:
        return "COMPARE_GREATER_OPERATION";
    default:
        return "OPERATION";
    }
}

string_t ternary_operation_type_string(ternary_operation_type_t ternary_operation_type)
{
    switch (ternary_operation_type)
    {
    case WHERE_OPERATION:
        return "WHERE_OPERATION";
    default:
        return "OPERATION";
    }
}

string_t reduction_operation_type_string(reduction_operation_type_t reduction_operation_type)
{
    switch (reduction_operation_type)
    {
    case SUMMATION_OPERATION:
        return "SUMMATION_OPERATION";
    case MAXIMUM_OPERATION:
        return "MAXIMUM_OPERATION";
    default:
        return "OPERATION";
    }
}

string_t structure_operation_type_string(structure_operation_type_t structure_operation_type)
{
    switch (structure_operation_type)
    {
    case EXPAND_OPERATION:
        return "EXPAND_OPERATION";
    case PERMUTE_OPERATION:
        return "PERMUTE_OPERATION";
    case RESHAPE_OPERATION:
        return "RESHAPE_OPERATION";
    case SLICE_OPERATION:
        return "SLICE_OPERATION";
    case PADDING_OPERATION:
        return "PADDING_OPERATION";
    case IMAGE_TO_COLUMN_OPERATION:
        return "IMAGE_TO_COLUMN_OPERATION";
    case COLUMN_TO_IMAGE_OPERATION:
        return "COLUMN_TO_IMAGE_OPERATION";
    default:
        return "OPERATION";
    }
}

string_t creation_operation_type_string(creation_operation_type_t creation_operation_type)
{
    switch (creation_operation_type)
    {
    case EMPTY_OPERATION:
        return "EMPTY_OPERATION";
    case ZEROES_OPERATION:
        return "ZEROES_OPERATION";
    case ONES_OPERATION:
        return "ONES_OPERATION";
    case UNIFORM_OPERATION:
        return "UNIFORM_OPERATION";
    case NORMAL_OPERATION:
        return "NORMAL_OPERATION";
    case ARANGE_OPERATION:
        return "ARANGE_OPERATION";
    case FROM_OPERATION:
        return "FROM_OPERATION";
    case COPY_OPERATION:
        return "COPY_OPERATION";
    default:
        return "OPERATION";
    }
}
