/**
 * @file view.c
 * @brief The view defines an interpretation of the underlying storage 
 *        used to represent a tensor. File contains operations to create, 
 *        manipulate, and describe the view.
 */

#include <view.h>
#include <stdio.h>
#include <string.h>

#pragma region VIEW_HELPERS

/**
 * @brief Given the shape and rank of a tensor that is contiguous in memory and stored in
 *        row-major format, find the associated strides. See unit tests in `test_view.c`
 *        for examples.
 *        Reference: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py
 * @param[out] strides The number of elements to skip in memory to reach the next 
 *                     element in a specific dimension of the tensor. The strides 
 *                     should be preallocated and same size as shape.
 * @param[in] shape An array of size rank representing the dimensions of the tensor.
 * @param[in] rank The order of the tensor. Gives the number of elements in shape.
 * @return Error if `shape` or `strides` is NULL.
 *         Error if `rank` does not satisfy 0 <= rank <= MAX_DIM.
 *         Error if any of the dimensions in shape are 0.
 *         NULL if the strides were successfully acquired.
 */
nw_error_t *strides_from_shape(int64_t *strides, const int64_t *shape, int64_t rank)
{
    CHECK_NULL_ARGUMENT(strides, "strides");
    CHECK_NULL_ARGUMENT(shape, "shape");

    if (rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK, string_create("rank %ld must be less than or equal to %d.", rank, (int) MAX_RANK), NULL);
    }

    for (int64_t i = 0; i < rank; ++i)
    {
        if (!shape[i])
        {
            return ERROR(ERROR_SHAPE,
                         string_create("all shape dimensions must be greater than 0."),
                         NULL);
        }

        if (!i)
        {
            strides[rank - (i + 1)] = 1;
        }
        else
        {
            strides[rank - (i + 1)] = shape[rank - i] * strides[rank - i];
        }
    }

    for (int64_t i = 0; i < rank; ++i)
    {
        if (shape[i] == 1)
        {
            strides[i] = 0;
        }
    }

    return NULL;
}

int64_t dimension_to_index(int64_t dimension, int64_t rank)
{
    return (dimension < 0) ? (rank + dimension) : dimension;
}

static bool_t axis_in_range(const int64_t *axis, int64_t length, int64_t rank)
{
    if (axis)
    {
        for (int64_t i = 0; i < length; ++i)
        {
            int64_t index = dimension_to_index(axis[i], rank);
            if (index < 0 || index >= rank)
            {
               return false;
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

/**
 * @brief Determine if an integer array contains a given value.
 * @param array The collection of integers being queried.
 * @param length The length of `array`.
 * @param value The value being searched for in the array.
 * @return True if `value` is present in the `array` and false otherwise.
 */
static bool_t array_contains(const int64_t *array, int64_t length, int64_t value)
{
    if (array)
    {
        for (int64_t i = 0; i < length; ++i)
        {
            if (value == array[i])
            {
                return true;
            }
        }
    }

    return false;
}

/**
 * @brief Given two arrays, determine if both have the same elements in the same positions. 
 * @param[in] array_a An array of size `length_a`.
 * @param[in] length_a Gives number of elements in `array_a`.
 * @param[in] array_b An array of size `length_b`.
 * @param[in] length_b Gives number of elements in `array_b`.
 * @return False if either shapes are NULL, ranks are not equal, or shape dimensions at a common index are not equal. 
 *         True if ranks are equal and shape dimensions are equal or both are NULL.
 */
static bool_t arrays_equal(const int64_t *array_a, int64_t length_a, const int64_t *array_b, int64_t length_b)
{
    if (length_a != length_b || (!array_a && array_b) || (array_a && !array_b))
    {
        return false;
    }

    if (!array_a && !array_b)
    {
        return true;
    }

    for (int64_t i = 0; i < length_a; ++i)
    {
        if (array_a[i] != array_b[i])
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief Take the product of all elements in an array,
 * @param[in] array A collection of integers being multiplied together.
 * @param[in] length Number of elements in `array`.
 * @return The product of elements in the array.
 */
static int64_t array_product(const int64_t *array, int64_t length)
{
    int64_t product = 1;

    if (!array)
    {
        return product;
    }

    for (int64_t i = 0; i < length; ++i)
    {
        product *= array[i];
    }
    
    return product;
}

static bool_t is_expandable(const int64_t *original_shape, int64_t original_rank, const int64_t *expanded_shape, int64_t expanded_rank)
{
    if (!original_shape || !expanded_shape || expanded_rank < original_rank)
    {
        return false;
    }

    for (int64_t i = 1; i < expanded_rank + 1; ++i)
    {
        if (original_rank >= i && original_shape[original_rank - i] != expanded_shape[expanded_rank - i] && original_shape[original_rank - i] != 1)
        {
            return false;
        }
    }

    return true;
}

#pragma endregion VIEW_HELPERS

/**
 * @brief Dynamically memory allocate and initialize a view. 
 *        Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html       
 * @param[out] view Pointer to allocated view memory. 
 * @param[in] offset The offset in the underlying storage in terms of number of 
 *                   storage elements (not bytes).
 *                   Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html
 * @param[in] rank The rank of the tensor (length of shape). Rank needs to be
 *                 in the closed interval `[0, MAX_RANK]`.
 *                 Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.ndimension.html
 * @param[in] shape The dimensions of the tensor. Must not be NULL. 
 *                  Contents is copied to dynamically memory allocated 
 *                  shape member in view.
 *                  Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.size.html
 * @param[in] strides The strides are the jumps necessary to go from one element 
 *                    to the next one in storage along each dimension (not bytes).
 *                    Contents is copied to dynamically memory allocated strides 
 *                    member in view if not NULL otherwise the allocated strides will be 
 *                    initialized under the assumption that the tensor is contiguous 
 *                    and in row major format.
 *                    Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html
 * @return Error if failed to dynamically allocate memory for view or any of its members.
 *         Error if `rank` does not satisfy `0 <= rank <= MAX_RANK`.
 *         Error of the contiguous tensor strides failed to be computed.
 *         Error if a dimension in shape is less than or equal to 0.
 *         NULL if view was successfully dynamically memory allocated and initialized.
 */
nw_error_t *view_create(view_t **view, int64_t offset, int64_t rank, const int64_t *shape, const int64_t *strides)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(shape, "shape");

    if (rank < 0 || rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK, string_create("rank must be in the interval [0, %d].", rank, (int) MAX_RANK), NULL);
    }

    for (int64_t i = 0; i < rank; ++i)
    {
        if (shape[i] <= 0)
        {
            return ERROR(ERROR_SHAPE, string_create("all tensor dimensions must be greater than 0 dimension <= shape dimesion."), NULL);
        }
    }

    nw_error_t *error = NULL;
    size_t size = rank * sizeof(int64_t);

    // View
    *view = (view_t *) malloc(sizeof(view_t));
    if (!*view)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate view of size %zu.", sizeof(view_t)), NULL);
        goto cleanup;
    } 

    // Initialize
    (*view)->offset = offset;
    (*view)->rank = rank;
    (*view)->shape = NULL;
    (*view)->strides = NULL;

    // Dynamically allocate memory for shape, strides.
    // If rank is 0, this should return a reference to a block of memory of size 0.

    (*view)->shape = (int64_t *) malloc(size);
    if (!(*view)->shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    (*view)->strides = (int64_t *) malloc(size);
    if (!(*view)->strides)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    // Copy
    // Don't need to copy for scalar tensors.
    if (rank)
    {
        memcpy((*view)->shape, shape, size);
        if (strides)
        {
            memcpy((*view)->strides, strides, size);
        }
        else
        {
            error = strides_from_shape((*view)->strides, shape, rank);
            if (error)
            {
                error = ERROR(ERROR_INITIALIZATION, string_create("failed to initialize strides."), error);
                goto cleanup;
            }
        }
    }

    return error;

cleanup:

    view_destroy(*view);

    return error;
}

/**
 * @brief Free memory dynamically allocated for view and its members.
 * @param[in] view The view instance to be freed.
 */
void view_destroy(view_t *view)
{
    if (view)
    {
        free(view->shape);
        free(view->strides);
        free(view);
    }
}

nw_error_t *view_create_contiguous(view_t **view, const int64_t *shape, int64_t rank)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = view_create(view, 0, rank, shape, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    return error;
}

/**
 * @brief Copy contents of a source view to a destination view.
 * @param[in] source_view The view whose contents is being copied. Must not be NULL.
 * @param[out] destination_view The view to write the copied contents to. Must not be NULL.
 * return Error if `source_view` or `destination_view` is NULL.
 *        Error if destination view could not be allocated and initialized.
 *        NULL if copy was successful.
 */
nw_error_t *view_copy(const view_t *source_view, view_t **destination_view)
{
    CHECK_NULL_ARGUMENT(source_view, "source_view");
    CHECK_NULL_ARGUMENT(destination_view, "destination_view");

    nw_error_t *error = NULL;

    error = view_create(destination_view, source_view->offset, source_view->rank, 
                        source_view->shape, source_view->strides);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }        

    return error;
}

/**
 * @brief Given the view of a tensor determine it is contiguous. 
 *        Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html
 *        Reference: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py 
 * @param[in] view The view of a tensor being checked for contiguous property. 
 * @param[out] is_contiguous True if tensor is contiguous, False if the tensor is not contiguous.
 * @return Error if argument if `view` or `is_contiguous` is NULL.
 *         Error if strides failed to initialize. 
 *         NULL if contiguous check was successful.
 */
nw_error_t *view_is_contiguous(const view_t *view, bool_t *is_contiguous)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(is_contiguous, "is_contiguous");

    nw_error_t *error = NULL;
    int64_t strides[view->rank];

    if (view->offset)
    {
        *is_contiguous = false;
        return error;
    }

    error = strides_from_shape(strides, view->shape, view->rank);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize strides."), error);
    }

    for (int64_t i = 0; i < view->rank; ++i)
    {
        if (view->strides[i] != strides[i] && view->shape[i] != 1)
        {
            *is_contiguous = false;
            return error;
        }
    }

    *is_contiguous = true;

    return error;
}

nw_error_t *view_permute(const view_t *original_view, view_t **permuted_view, const int64_t *axis, int64_t length)
{
    CHECK_NULL_ARGUMENT(original_view, "original_view");
    CHECK_NULL_ARGUMENT(permuted_view, "permuted_view");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NEGATIVE_ARGUMENT(length, "length");
    CHECK_UNIQUE(axis, length, "axis");

    nw_error_t *error = NULL;

    if (length != original_view->rank)
    {
        return ERROR(ERROR_AXIS, string_create("length of axis much match rank of tensor."), NULL);
    }

    if (!axis_in_range(axis, length, original_view->rank))
    {
        return ERROR(ERROR_AXIS, string_create("axis dimensions out of range."), NULL);
    }

    error = view_copy(original_view, permuted_view);
    if (error)
    {
        return ERROR(ERROR_COPY, string_create("failed to copy view."), error);
    }
    
    for (int64_t i = 0; i < length; ++i)
    {
        int64_t j = dimension_to_index(axis[i], original_view->rank);
        (*permuted_view)->shape[i] = original_view->shape[j];
        (*permuted_view)->strides[i] = original_view->strides[j];
    }
    
    return error;
}

/**
 * @brief Given a view of a tensor that has been reduced without keeping
 *        the dimensions, recover the dimensions of the tensor such that
 *        the recovered shape, strides, and mask are equal to the shape, strides, and mask of the
 *        tensor if it was reduced with keeping the dimensions respectively.
 * @param[in] reduced_view The view of a tensor reduced along dimensions specified by `axis` without keeping dimension.
 * @param[out] recovered_view The view of the reduced tensor if it was reduced with keeping dimension instead of without keeping dimension.
 * @param[in] axis The indicies of the dimensions of the tensor that were reduced and removed.
 * @param[in] rank The number of elements in `axis`.
 * @return Error if `reduced_view`, `recovered_view`, or `axis` is NULL.
 *         Error if not all dimension indices in `axis` are unique.
 *         Error if negative argument received for `length`.
 *         NULL if the view was recovered successfully.
 */
nw_error_t *view_recover_dimensions(const view_t *reduced_view, view_t **recovered_view, const int64_t *axis, int64_t length)
{
    CHECK_NULL_ARGUMENT(reduced_view, "reduced_view");
    CHECK_NULL_ARGUMENT(recovered_view, "recovered_view");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NEGATIVE_ARGUMENT(length, "length");
    CHECK_UNIQUE(axis, length, "axis");

    if (!axis_in_range(axis, length, reduced_view->rank + length))
    {
        return ERROR(ERROR_AXIS, string_create("axis dimensions out of range."), NULL);
    }

    nw_error_t *error = NULL;
    int64_t recovered_rank = reduced_view->rank + length;
    int64_t recovered_shape[recovered_rank];
    int64_t recovered_strides[recovered_rank];
    int64_t recovered_offset = reduced_view->offset;
    int64_t k = 0;

    for (int64_t i = 0; i < recovered_rank; ++i)
    {
        if (array_contains(axis, length, i) || array_contains(axis, length, i - recovered_rank))
        {
            recovered_shape[i] = 1;
            recovered_strides[i] = 0;
        }
        else
        {
            if (k >= reduced_view->rank)
            {
                return ERROR(ERROR_RANK, string_create("error index %ld out of range of reduced rank %ld.", k, reduced_view->rank), NULL);
            }

            recovered_shape[i] = reduced_view->shape[k];
            recovered_strides[i] = reduced_view->strides[k];
            ++k;
        }
    }

    error = view_create(recovered_view, recovered_offset, recovered_rank, recovered_shape, recovered_strides);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    return error;
}

/**
 * @brief Given a view of a tensor and collection of dimensions to reduce, 
 *        find the resultant view of the tensor after reduction is applied 
 *        over the selected dimensions.
 * @param[in] original_view The view of the tensor before reduction.
 * @param[out] reduced_view The view of the reduced tensor.
 * @param[in] axis An array of inidices of tensor dimensions to reduce.
 * @param[in] rank The number of indices in `axis`.
 * @param[in] keep_dimensions A flag to indicate whether the dimensions are retained (true)
 *                            after reduction or removed (false).
 * @return Error if `original_view`, `reduced_view`, or `axis` are NULL.
 *         Error if not all dimension indices in `axis` are unique or in range of tensor rank.
 *         Error if view failed to be allocated and initialized.
 *         NULL if reduced view was successfully determined.
 */
nw_error_t *view_reduce(const view_t *original_view, view_t **reduced_view, const int64_t *axis, int64_t length, bool_t keep_dimensions)
{
    CHECK_NULL_ARGUMENT(original_view, "original_view");
    CHECK_NULL_ARGUMENT(reduced_view , "reduced_view");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NEGATIVE_ARGUMENT(length, "length");
    CHECK_UNIQUE(axis, length, "axis");

    if (length > original_view->rank)
    {
        return ERROR(ERROR_RANK, string_create("number of dimensions being reduced %ld is greater than rank of tensor %ld.", length, original_view->rank), NULL);
    }

    if (!axis_in_range(axis, length, original_view->rank))
    {
        return ERROR(ERROR_AXIS, string_create("axis dimensions out of range."), NULL);
    }

    nw_error_t *error = NULL;
    int64_t reduced_rank = keep_dimensions ? original_view->rank : original_view->rank - length;
    int64_t reduced_strides[reduced_rank];
    int64_t reduced_shape[reduced_rank];
    int64_t reduced_offset = 0;
    int64_t k = reduced_rank - 1;
    int64_t stride = 1;

    for (int64_t i = original_view->rank - 1; i >= 0; i--)
    {
        if (array_contains(axis, length, i) || array_contains(axis, length, i - original_view->rank))
        {
            if (keep_dimensions)
            {
                reduced_shape[k] = 1;
                reduced_strides[k] = 0;
                k--;
            }
        }
        else
        {
            reduced_shape[k] = original_view->shape[i];
            if (!original_view->strides[i])
            {
                reduced_strides[k] = 0;
            }
            else
            {
                reduced_strides[k] = stride;
                stride *= reduced_shape[k];
            }
            k--;
        }
    }

   error = view_create(reduced_view, reduced_offset, reduced_rank, reduced_shape, reduced_strides);
   if (error)
   {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
   }

    return error;
}

/**
 * @brief Given the view of a tensor, compute the required
 *        number of tensor data elements that need to be stored in memory.
 * @param[in] view The view of the tensor.
 * @param[out] size The number of elements that are required to be stored in 
    *               memory to represent the tensor.
 * @return Error if `view` or `size` is NULL.
 *         Error if any of the dimensions are zero.
 *         NULL if `size` is computed successfully.
 */
nw_error_t *view_physical_size(const view_t *view, int64_t *size)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(view->strides, "view->strides");
    CHECK_NULL_ARGUMENT(size, "size");

    *size = array_product(view->shape, view->rank);

    for (int64_t i = 0; i < view->rank; ++i)
    {
        if (!view->strides[i])
        {
            if (view->shape[i])
            {
                *size /= view->shape[i];
            }
            else
            {
                return ERROR(ERROR_SHAPE, string_create("all dimensions of the tensor must be greater than 0."), NULL);
            }
        }
    }

    return NULL;
}

/**
 * @brief Given the view of a tensor, find the logical number of elements in the tensor.
 *        Note that this is just the percieved number of elements in the tensor and 
 *        may not be equal to the number of elements physically stored in memory.
 *        For the number of elements stored in memory use `view_physical_size`. 
 *        Reference: https://pytorch.org/docs/stable/generated/torch.numel.html
 * @param[in] view The view of the tensor.
 * @param[out] size The logical number of elements in the tensor.
 * @return Error if `view` or `size` is NULL.
 *         NULL if `size` is successfully computed.
 */
nw_error_t *view_logical_size(const view_t *view, int64_t *size)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(size, "size");

    *size = array_product(view->shape, view->rank);

    return NULL;
}

bool_t view_shapes_equal(const view_t *view_a, const view_t *view_b)
{
    return view_a && view_b && arrays_equal(view_a->shape, view_a->rank, view_b->shape, view_b->rank);
}

bool_t view_has_shape(const view_t *view, const int64_t *shape, int64_t rank)
{
    return view && shape && arrays_equal(view->shape, view->rank, shape, rank);
}

/**
 * @brief Given the view of a tensor, and the target shape and rank to expand the tensor to,
 *        get the resultant view of the tensor expanded to the target shape.   
 *        Reference: See broadcasting rules at https://numpy.org/doc/stable/user/basics.broadcasting.html.
 * @param[in] original_view The view of the original tensor being expanded.
 * @param[out] expanded_view The view of the tensor expanded to the target shape.
 * @param[in] shape An array of size `rank` representing the dimensions of the target expanded tensor.
 * @param[in] rank The order of the expanded tensor. Gives the number of elements in `shape`.
 * @return NULL if expansion was successful. 
 *         Error if `original_view`, `expanded_view`, or `shape` are NULL.
 *         Error if tensor cannot be expanded to target shape.
 */
nw_error_t *view_expand(const view_t *original_view, view_t **expanded_view, const int64_t *shape, int64_t rank)
{
    CHECK_NULL_ARGUMENT(original_view, "original_view");
    CHECK_NULL_ARGUMENT(expanded_view, "expanded_view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NEGATIVE_ARGUMENT(rank, "rank");

    if (!is_expandable(original_view->shape, original_view->rank, shape, rank))
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand view."), NULL);
    }

    nw_error_t *error = NULL;
    int64_t strides[rank];

    for (int64_t i = 1; i < rank + 1; ++i)
    {   
        int64_t original_index = original_view->rank - i;
        int64_t broadcast_index = rank - i;

        if (i > original_view->rank || original_view->shape[original_index] == 1)
        {
            strides[broadcast_index] = 0; 
        }
        else if (original_view->shape[original_index] == shape[broadcast_index])
        {
            strides[broadcast_index] = original_view->strides[original_index];
        }
        else
        {
            return ERROR(ERROR_EXPAND, string_create("failed to expand view."), NULL);
        }
    }

    error = view_create(expanded_view, original_view->offset, rank, shape, strides);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }


    return error;
}

/**
 * @brief Given the views of two tensors being combined via an elementwise binary operation, find
 *        the associated shape and rank to expand both tensors to perform the operation.   
 *        Reference: See broadcasting rules at https://numpy.org/doc/stable/user/basics.broadcasting.html.
 * @param[in] view_a The view of the first tensor operand.
 * @param[in] view_b The view of the second tensor operand.
 * @param[out] shape An array of size `rank` representing the dimensions of the expanded tensor.
 * @param[out] rank The order of the expanded tensor. Gives the number of elements in `shape`.
 * @return NULL if operation was successful.
 *         Error if any argument is NULL.
 *         Error if view shapes cannot be broadcasted together.
 */
nw_error_t *view_broadcast(const view_t *view_a, const view_t *view_b, int64_t **shape, int64_t *rank)
{
    CHECK_NULL_ARGUMENT(view_a, "view_a"); 
    CHECK_NULL_ARGUMENT(view_b, "view_b"); 
    CHECK_NULL_ARGUMENT(shape, "shape"); 
    CHECK_NULL_ARGUMENT(rank, "rank"); 

    *rank = MAX(view_a->rank, view_b->rank);
    size_t size = *rank * sizeof(int64_t);
    *shape = (int64_t *) malloc(size);
    if (!*shape)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    for (int64_t i = 1; i < *rank + 1; ++i)
    {
        int64_t index_a = view_a->rank - i;
        int64_t index_b = view_b->rank - i;
        int64_t index = *rank - i;
        if (i > view_a->rank || (i <= view_b->rank && view_a->shape[index_a] == 1))
        {
            (*shape)[index] = view_b->shape[index_b];
        } 
        else if (i > view_b->rank || view_a->shape[index_a] == view_b->shape[index_b] || view_b->shape[index_b] == 1)
        {
            (*shape)[index] = view_a->shape[index_a];
        }
        else
        {
            free(*shape);
            *shape = NULL;
            return ERROR(ERROR_BROADCAST, string_create("failed to broadcast shapes."), NULL);
        }
    }

    return NULL;
}

/**
 * @brief Given the views of two tensors being combined via a matrix multiplication operation, find
 *        the associated shapes and rank to expand the operands to.   
 * @param[in] view_a The view of the first tensor operand.
 * @param[in] view_b The view of the second tensor operand.
 * @param[out] shape_a An array of size `rank` representing the target dimensions to expand the first tensor operand to.
 * @param[out] shape_b An array of size `rank` representing the target dimensions to expand the second tensor operand to.
 * @param[out] rank The order of the expanded tensors. Gives the number of elements in `shape_a` and `shape_b`.
 * @return NULL if operation was successful. An error if any pointers are NULL or shapes cannot be broadcasted together.
 *         See broadcasting rules at https://pytorch.org/docs/stable/generated/torch.matmul.html.
 */
nw_error_t *view_broadcast_matrix_multiplication(const view_t *view_a, const view_t *view_b, int64_t **shape_a, int64_t **shape_b, int64_t *rank)
{
    CHECK_NULL_ARGUMENT(view_a, "view_a"); 
    CHECK_NULL_ARGUMENT(view_b, "view_b"); 
    CHECK_NULL_ARGUMENT(shape_a, "shape_a"); 
    CHECK_NULL_ARGUMENT(shape_b, "shape_b"); 

    nw_error_t *error = NULL;
    *shape_a = NULL;
    *shape_b = NULL;
    *rank = MAX(view_a->rank, view_b->rank);
    size_t size = *rank * sizeof(int64_t);

    if (view_a->rank < 2 || view_b->rank < 2)
    {
        error = ERROR(ERROR_RANK, string_create("operands must be atleast rank 2 or above."), NULL);
        goto cleanup;
    }

    *shape_a = (int64_t *) malloc(size);
    if (!*shape_a)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    *shape_b = (int64_t *) malloc(size);
    if (!*shape_b)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    for (int64_t i = 1; i < *rank + 1; ++i)
    {
        int64_t index_a = view_a->rank - i;
        int64_t index_b = view_b->rank - i;
        int64_t index = *rank - i;
        if (i < 3)
        {
            (*shape_a)[index] = view_a->shape[index_a];
            (*shape_b)[index] = view_b->shape[index_b];
            continue;
        }

        if (i > view_a->rank || (i <= view_b->rank && view_a->shape[index_a] == 1))
        {
            (*shape_a)[index] = view_b->shape[index_b];
            (*shape_b)[index] = view_b->shape[index_b];
        } 
        else if (i > view_b->rank || view_a->shape[index_a] == view_b->shape[index_b] || view_b->shape[index_b] == 1)
        {
            (*shape_a)[index] = view_a->shape[index_a];
            (*shape_b)[index] = view_a->shape[index_a];
        }
        else
        {
            error = ERROR(ERROR_BROADCAST, string_create("failed to broadcast shapes."), NULL);
            goto cleanup;
        }
    }

    return error;

cleanup:

    free(*shape_a);
    free(*shape_b);
    *shape_a = NULL;
    *shape_b = NULL;

    return error;
}

nw_error_t *view_matrix_multiplication(const view_t *view_a, const view_t *view_b, view_t **view_c)
{
    CHECK_NULL_ARGUMENT(view_a, "view_a");
    CHECK_NULL_ARGUMENT(view_b, "view_b");
    CHECK_NULL_ARGUMENT(view_c, "view_c");

    nw_error_t *error = NULL;
    int64_t rank = MAX(view_a->rank, view_b->rank);
    int64_t shape[rank];

    if (view_a->rank < 2 || view_b->rank < 2)
    {
        return ERROR(ERROR_RANK, string_create("operands must be atleast rank 2 or above."), NULL);
    }

    if (view_a->shape[rank - 1] != view_b->shape[rank - 2])
    {
        return ERROR(ERROR_SHAPE, 
                     string_create("number of columns in first operand %ld not equal to number of rows in second operand %ld.", 
                     view_a->shape[rank - 1], view_b->shape[rank - 2]), NULL);
    }

    for (int64_t i = 1; i < rank + 1; ++i)
    {
        int64_t j = rank - i;
        if (i == 1)
        {
            shape[j] = view_b->shape[j];
        }
        else if (i == 2)
        {
            shape[j] = view_a->shape[j];
        }
        else
        {
            if (view_a->shape[j] == view_b->shape[j])
            {
                shape[j] = view_a->shape[j];
            }
            else
            {
                return ERROR(ERROR_SHAPE,
                             string_create("dimension in first operand %ld not equal to dimension in second operand %ld.",
                             view_a->shape[j], view_b->shape[j]), NULL);
            }
        }
    }

    error = view_create_contiguous(view_c, shape, rank);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    return error;
}

nw_error_t *view_reduce_axis(const view_t *original_view, 
                             const int64_t *broadcasted_shape, int64_t broadcasted_rank,
                             int64_t **axis_keep_dimension, int64_t *length_keep_dimension,
                             int64_t **axis_remove_dimension, int64_t *length_remove_dimension)
{
    CHECK_NULL_ARGUMENT(original_view, "original_view");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(axis_keep_dimension, "axis_keep_dimension");
    CHECK_NULL_ARGUMENT(axis_remove_dimension, "axis_remove_dimension");
    CHECK_NULL_ARGUMENT(length_keep_dimension, "length_keep_dimension");
    CHECK_NULL_ARGUMENT(length_remove_dimension, "length_remove_dimension");
    
    if (original_view->rank > MAX_RANK || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK, 
                     string_create("original rank %ld and broadcasted rank %ld must be less than or equal to %d.", 
                     original_view->rank, broadcasted_rank, (int) MAX_RANK),
                     NULL);
    }

    if (!is_expandable(original_view->shape, original_view->rank, broadcasted_shape, broadcasted_rank))
    {
        return ERROR(ERROR_BROADCAST,
                     string_create("cannot broadcast shapes."),
                     NULL);
    }


    for (int i = 0; i < 2; ++i)
    {
        if (i)
        {
            *axis_keep_dimension = (int64_t *) malloc((*length_keep_dimension) * sizeof(int64_t));
            if (!*axis_keep_dimension)
            {
                return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (*length_keep_dimension) * sizeof(int64_t)), NULL);
            }
            *axis_remove_dimension = (int64_t *) malloc((*length_remove_dimension) * sizeof(int64_t));
            if (!*axis_remove_dimension)
            {
                free(*axis_keep_dimension);
                *axis_keep_dimension = NULL;
                return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (*length_remove_dimension) * sizeof(int64_t)), NULL);
            }
        }
        *length_keep_dimension = 0;
        *length_remove_dimension = 0;

        for (int64_t j = 0; j < broadcasted_rank; ++j)
        {
            if (original_view->rank >= (j + 1))
            {
                if (original_view->shape[original_view->rank - (j + 1)] != 
                    broadcasted_shape[broadcasted_rank - (j + 1)])
                {
                    if (i)
                    {
                        (*axis_keep_dimension)[*length_keep_dimension] = broadcasted_rank - (j + 1);
                    }
                    ++(*length_keep_dimension);
                }
            }
            else
            {
                if (i)
                {
                    (*axis_remove_dimension)[*length_remove_dimension] = broadcasted_rank - (j + 1);
                }
                ++(*length_remove_dimension);
            }
        }
    }

    return NULL;
}
