#include <view.h>
#include <stdio.h>

/**
 * @brief Allocate and initialize view. 
 * @param view Address to view pointer to allocate memory for. 
 * @param offset The memory offset to the first element. 
 * @param rank The rank of the tensor.
 * @param shape The dimensions of the tensor. Must not be NULL. Contents is copied to shape member in view.
 * @param strides The memory strides to traverse between elements of a tensor along each dimension. 
 *                Contents is copied to strides member in view if not NULL otherwise strides will be initialized
 *                assuming tensor is contiguous and in row major format.
 * @return Memory allocation error if failed to allocate memory for a member in the view.
 */
error_t *view_create(view_t **view, uint32_t offset, uint32_t rank, const uint32_t *shape, const uint32_t *strides)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(shape, "shape");

    if (rank < 1 || rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank %u must be between 1 and %d.",
                     (unsigned int) rank, (int) MAX_RANK), NULL);
    }

    // View
    *view = (view_t *) malloc((size_t) sizeof(view_t));
    if (view == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view of size %lu.",
                     (unsigned long) sizeof(view_t)), NULL);
    } 

    // Shape
    (*view)->shape = (uint32_t *) malloc((size_t) (rank * sizeof(uint32_t)));
    if ((*view)->shape == NULL)
    {
        free(*view);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view->shape of size %lu.",
                     (unsigned long) (rank * sizeof(uint32_t))), NULL);
    }

    // Strides
    (*view)->strides = (uint32_t *) malloc((size_t) (rank * sizeof(uint32_t)));
    if ((*view)->strides == NULL)
    {
        free(*view);
        free((*view)->shape);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view->strides of size %lu.",
                     (unsigned long) (rank * sizeof(uint32_t))), NULL);
    }

    // Copy
    memcpy((void *) ((*view)->shape), (const void *) shape, (size_t) (rank * sizeof(uint32_t)));
    if (strides != NULL)
    {
        memcpy((void *) ((*view)->strides), (const void *) strides, (size_t) (rank * sizeof(uint32_t)));
    }
    else
    {
        error_t *error = strides_from_shape((*view)->strides, shape, rank);
        if (error != NULL)
        {
            free((*view)->strides);
            free((*view)->shape);
            free(*view);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create strides from shape."),
                         error);
        }
    }

    // Initialize
    (*view)->offset = offset;
    (*view)->rank = rank;

    return NULL;
}

/**
 * @brief Free memory allocated for view.
 * @param view The view instance to be freed.
 */
void view_destroy(view_t *view)
{
    if (view == NULL)
    {
        return;
    }

    free(view->shape);
    free(view->strides);
    free(view);
}


/**
 * @brief Determine if tensor is contiguous in memory.
 * @param shape The dimensions of the tenors. 
 * @param rank  The rank of the tensor. 
 * @param strides The memory strides of the tensor.
 * @return True if the tensor memory is contiguous and False if it isn't. 
 */
bool_t is_contiguous(const uint32_t *shape, uint32_t rank, const uint32_t *strides)
{
    if (shape == NULL || strides == NULL)
    {
        return false;
    }

    if (rank < 1 || rank > MAX_RANK)
    {
        return false;
    }

    uint32_t contiguous_strides[rank];    
    error_t *error = strides_from_shape(contiguous_strides, shape, rank);
    if (error != NULL)
    {
        error_destroy(error);
        return false;
    }

    for (uint32_t i = 0; i < rank; i++)
    {
        if (strides[i] != contiguous_strides[i] && shape[i] != 1)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief Permute the dimensions of a tensor.
 * @param original_shape The original dimensions. 
 * @param original_rank The size of the original_shape and original_strides array.
 * @param original_strides The original memory strides.
 * @param permuted_shape The reordered dimensions.
 * @param permuted_rank The size of the permuted_shape and permute_strides array.
 * @param permuted_strides The reordered memory strides.
 * @param axis Array containing the order of dimensions.
 * @param length Number of elements in axis.
 * @return Rank conflict error if the permute shape array is not the 
 *         same size as the original shape array or axis length array.
 */
error_t *permute(const uint32_t *original_shape,
                 uint32_t original_rank,
                 const uint32_t *original_strides, 
                 uint32_t *permuted_shape,
                 uint32_t permuted_rank,
                 uint32_t *permuted_strides,
                 const uint32_t *axis,
                 uint32_t length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides, "original_strides");
    CHECK_NULL_ARGUMENT(permuted_shape, "permuted_shape");
    CHECK_NULL_ARGUMENT(permuted_strides, "permuted_strides");
    CHECK_NULL_ARGUMENT(axis, "axis");

    if (original_rank != permuted_rank || original_rank != length || length != permuted_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with original rank %u, permuted rank %u and axis length %u.",
                     (unsigned int) original_rank, (unsigned int) permuted_rank, (unsigned int) length), NULL);
    }

    if (original_rank < 1 || original_rank > MAX_RANK || 
        permuted_rank < 1 || permuted_rank > MAX_RANK ||
        length < 1 || length > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("original rank %u, permuted rank %u and axis length %u must be between 1 and %d.",
                     (unsigned int) original_rank, (unsigned int) permuted_rank, (unsigned int) length, (int) MAX_RANK),
                     NULL);
    }
    
    for (uint32_t i = 0; i < length; i++)
    {
        uint32_t dimension = axis[i];
        if (dimension < original_rank)
        {
            permuted_shape[i] = original_shape[dimension];
            permuted_strides[i] = original_strides[dimension];
        }
        else
        {
            return ERROR(ERROR_PERMUTE,
                         string_create("axis dimension %u out of range of rank %u.", 
                         (unsigned int) dimension, (unsigned int) original_rank), NULL);
        }
    }
    
    return NULL;
}

typedef struct pair_t
{
    uint32_t index;
    uint32_t value;
} pair_t;
    
static int compare(const void *a, const void *b)
{
    pair_t *pair_a = (pair_t *) a;
    pair_t *pair_b = (pair_t *) b;

    return (pair_a->value - pair_b->value);
}

error_t *reverse_permute(const uint32_t *axis, uint32_t rank, uint32_t *reverse_axis)
{
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(reverse_axis, "reverse_axis");

    if (rank < 1 || rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank %u must be between 1 and %d.",
                     (unsigned int) rank, (int) MAX_RANK), NULL);
    }

    pair_t *new_axis = (pair_t *) malloc((size_t) (rank * sizeof(pair_t)));
    if (new_axis == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate new axis of size %lu bytes.",
                     (unsigned long) (rank * sizeof(pair_t))), NULL);
    }

    for (uint32_t i = 0; i < rank; i++)
    {
        new_axis[i].index = i;
        new_axis[i].value = axis[i];
    }

    qsort((void *) new_axis, (size_t) rank, sizeof(pair_t), compare);

    for (uint32_t i = 0; i < rank; i++)
    {
        reverse_axis[i] = new_axis[i].index;
    }

    free(new_axis);

    return NULL;
}

error_t *reduce_recover_dimensions(const uint32_t *original_shape,
                                   uint32_t original_rank, 
                                   const uint32_t *original_strides,
                                   uint32_t *reduced_shape, 
                                   uint32_t reduced_rank,
                                   uint32_t *reduced_strides,
                                   const uint32_t *axis,
                                   uint32_t rank)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides , "original_strides");
    CHECK_NULL_ARGUMENT(reduced_shape, "reduced_shape");
    CHECK_NULL_ARGUMENT(reduced_strides, "reduced_strides");
    CHECK_NULL_ARGUMENT(axis, "axis");

    if (reduced_rank != original_rank + rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with original rank %u, reduced rank %u and axis length %u.",
                     (unsigned int) original_rank, (unsigned int) reduced_rank, (unsigned int) rank), NULL);
    }

    if (original_rank < 1 || original_rank > MAX_RANK || 
        reduced_rank < 1 || reduced_rank > MAX_RANK ||
        rank < 1 || rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("original rank %u, reduced rank %u and axis length %u must be between 1 and %d.",
                     (unsigned int) original_rank, (unsigned int) reduced_rank, (unsigned int) rank, (int) MAX_RANK),
                     NULL);
    }

    for (uint32_t i = 0; i < rank; i++)
    {
        if (axis[i] >= reduced_rank)
        {
            return ERROR(ERROR_RANK_CONFLICT,
                         string_create("reduced rank %u must be greater than axis dimension %u.",
                         (unsigned int) reduced_rank, (unsigned int) axis[i]), NULL);
        }
    }

    uint32_t k = 0;
    for (uint32_t i = 0; i < reduced_rank; i++)
    {
        bool_t reduced = false;
        for (uint32_t j = 0; j < rank; j++)
        {
            if (axis[j] == i)
            {
                reduced_shape[i] = 1;
                reduced_strides[i] = 0;
                reduced = true;
                break;
            }
        }
    
        if (!reduced)
        {
            reduced_shape[i] = original_shape[k];
            reduced_strides[i] = original_strides[k];
            k++;
        }
    }

    return NULL;
}

error_t *reduce(const uint32_t *original_shape,
                uint32_t original_rank,
                const uint32_t *original_strides, 
                uint32_t *reduced_shape,
                uint32_t reduced_rank,
                uint32_t *reduced_strides,
                const uint32_t *axis,
                uint32_t rank,
                bool_t keep_dimensions)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides , "original_strides");
    CHECK_NULL_ARGUMENT(reduced_shape, "reduced_shape");
    CHECK_NULL_ARGUMENT(reduced_strides, "reduced_strides");
    CHECK_NULL_ARGUMENT(axis, "axis");

    if (keep_dimensions && original_rank != reduced_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with original rank %u and reduced rank %u.",
                     (unsigned int) original_rank, (unsigned int) reduced_rank), NULL);
    }

    if (!keep_dimensions && reduced_rank != original_rank - rank && !(reduced_rank == 1 && original_rank - rank == 0))
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with expected rank %u and reduced rank %u.", 
                     (unsigned int) (original_rank - rank), (unsigned int) reduced_rank), NULL);
    }

    if (rank < 1 || rank > original_rank || 
        original_rank < 1 || original_rank > MAX_RANK ||
        reduced_rank < 1 || reduced_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("original rank %u, permuted rank %u and axis length %u must be between 1 and %d and rank <= original rank.",
                     (unsigned int) original_rank, (unsigned int) reduced_rank, (unsigned int) rank, (int) MAX_RANK),
                     NULL);
    }

    for (uint32_t i = 0; i < rank; i++)
    {
        if (axis[i] >= original_rank)
        {
            return ERROR(ERROR_RANK_CONFLICT,
                         string_create("original rank %u must be greater than axis dimension %u.",
                         (unsigned int) original_rank, (unsigned int) axis[i]), NULL);
        }
    }

    uint32_t k = reduced_rank - 1;
    uint32_t stride = 1;
    for (uint32_t i = original_rank - 1; ; i--)
    {
        bool_t reduce_dimension = false;
        for (uint32_t j = 0; j < rank; j++)
        {
            if (axis[j] == i)
            {
                reduce_dimension = true;
                break;
            }
        }

        if (reduce_dimension && (keep_dimensions || (i == original_rank - 1 && original_rank - rank == 0)))
        {
            reduced_shape[k] = 1;
            if (k < reduced_rank - 1)
            {
                stride *= reduced_shape[k + 1];
            }
            reduced_strides[k] = 0;
            k--;
        }
        else if (!reduce_dimension)
        {
            reduced_shape[k] = original_shape[i];
            if (k < reduced_rank - 1)
            {
                stride *= reduced_shape[k + 1];
            }

            if (original_strides[i] == 0)
            {
                reduced_strides[k] = 0;
            }
            else
            {
                reduced_strides[k] = stride;
            }
            k--;
        }

        if (i == 0)
        {
            break;
        }
    }

    return NULL;
}

/**
 * @brief Given the shape and rank of two tensors, determine if both tensors have the same dimensions. 
 * @param x_shape An array of size x_rank representing the dimensions of a tensor.
 * @param x_rank The order of the tensor. Gives number of elements in x_shape.
 * @param y_shape An array of size y_rank representing the dimensions of a tensor.
 * @param y_rank The order of the tensor. Gives number of elements in y_shape.
 * @return False if either shapes are NULL, ranks are not equal, or shape dimensions are not equal. True otherwise.
 */
bool_t shapes_equal(const uint32_t *x_shape, uint32_t x_rank, const uint32_t *y_shape, uint32_t y_rank)
{
    if (x_shape == NULL || y_shape == NULL || x_rank != y_rank)
    {
        return false;
    }

    for (uint32_t i = 0; i < x_rank; i++)
    {
        if (x_shape[i] != y_shape[i])
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief Given the shape and rank of a tensor, find the total number of elements in the tensor. 
 * @param shape An array of size rank representing the dimensions of a tensor.
 * @param rank The order of the tensor. Gives number of elements in shape.
 * @return The size of the tensor. Returns 0 if shape is NULL.
 */
uint32_t shape_size(const uint32_t *shape, uint32_t rank)
{
    if (shape == NULL)
    {
        return 0;
    }

    uint32_t total = 0;
    for (uint32_t i = 0; i < rank; i++)
    {
        total = (i == 0) ? shape[i] : total * shape[i];
    }
    
    return total;
}

/**
 * @brief Given the shape and rank of a tensor that is contiguous and stored in
 *        row-major format, find the associated strides.
 * @param strides The number of elements to skip in memory to reach the next 
 *                 element in a specific dimension of the tensor. The strides 
 *                 should be preallocated and same size as shape.
 * @param shape An array of size rank representing the dimensions of the tensor.
 * @param rank The order of the tensor. Gives the number of elements in shape.
 * @return NULL if operation was successful. An error if strides or shape are NULL.
 */
error_t *strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank)
{
    CHECK_NULL_ARGUMENT(strides, "strides");
    CHECK_NULL_ARGUMENT(shape, "shape");

    if (rank < 1 || rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("rank %u must be between 1 and %d.", 
                     (unsigned int) rank, (int) MAX_RANK), NULL);
    }

    for (uint32_t i = 0; i < rank; i++)
    {
        if (i == 0)
        {
            strides[rank - (i + 1)] = 1;
        }
        else
        {
            strides[rank - (i + 1)] = shape[rank - i] * strides[rank - i];
        }
    }

    for (uint32_t i = 0; i < rank; i++)
    {
        if (shape[i] == 1)
        {
            strides[i] = 0;
        }
    }

    return NULL;
}

/**
 * @brief Given the shape, rank, and strides of a tensor, and the target shape and rank to broadcast the tensor to,
 *        get the strides required to broadcast the tensor to the target shape without copying memory.   
 * @param original_shape An array of size original_rank representing the dimensions of the original tensor being broadcasted.
 * @param original_rank The order of the original tensor. Gives the number of elements in original_shape.
 * @param original_strides The memory strides to traverse elements of the original tensor.
 * @param broadcasted_shape An array of size broadcasted_rank representing the dimensions of the target broadcasted tensor.
 * @param broadcasted_rank The order of the broadcasted tensor. Gives the number of elements in broadcasted_shape.
 * @param broadcasted_strides The memory strides to traverse elements of the broadcasted tensor.
 * @return NULL if operation was successful. An error if any pointers are NULL or shapes cannot be broadcasted together.
 *         See broadcasting rules at https://numpy.org/doc/stable/user/basics.broadcasting.html.
 */
error_t *broadcast_strides(const uint32_t *original_shape,
                           uint32_t original_rank,
                           const uint32_t *original_strides,
                           const uint32_t *broadcasted_shape,
                           uint32_t broadcasted_rank,
                           uint32_t *broadcasted_strides)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides, "original_strides");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(broadcasted_strides, "broadcasted_strides");

    if (original_rank < 1 || original_rank > MAX_RANK || 
        broadcasted_rank < 1 || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u and broadcasted rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (unsigned int) broadcasted_rank, (int) MAX_RANK), NULL);
    }

    if (!is_broadcastable(original_shape, original_rank, broadcasted_shape, broadcasted_rank))
    {
        string_t original_shape_string = uint32_array_to_string(original_shape, original_rank);
        string_t broadcasted_shape_string = uint32_array_to_string(broadcasted_shape, broadcasted_rank);
        error_t *error = ERROR(ERROR_BROADCAST,
                               string_create("cannot broadcast shape %s to shape %s.",
                               original_shape_string,
                               broadcasted_shape_string),
                               NULL);
        string_destroy(original_shape_string);
        string_destroy(broadcasted_shape_string);
        return error;
    }

    for (uint32_t i = 0; i < broadcasted_rank; i++)
    {   
        if ((i + 1) > original_rank || (original_shape[original_rank - (i + 1)] == 1))
        {
            broadcasted_strides[broadcasted_rank -  (i + 1)] = 0; 
        }
        else if (original_shape[original_rank - (i + 1)] == broadcasted_shape[broadcasted_rank - (i + 1)])
        {
            broadcasted_strides[broadcasted_rank - (i + 1)] = original_strides[original_rank - (i + 1)];
        }
        else
        {
            string_t original_shape_string = uint32_array_to_string(original_shape, original_rank);
            string_t broadcasted_shape_string = uint32_array_to_string(broadcasted_shape, broadcasted_rank);
            error_t *error = ERROR(ERROR_BROADCAST,
                                   string_create("cannot broadcast shape %s to shape %s.",
                                   original_shape_string,
                                   broadcasted_shape_string),
                                   NULL);
            string_destroy(original_shape_string);
            string_destroy(broadcasted_shape_string);
            return error;
        }
    }

    return NULL;
}

/**
 * @brief Given the shape, rank, and strides of two tensors being combined via an elementwise binary operation, find
 *        the associated shape and rank to broadcast both tensors to perform the operation.   
 * @param x_original_shape An array of size x_original_rank representing the dimensions of the original tensor being broadcasted.
 * @param x_original_rank The order of the original tensor. Gives the number of elements in x_original_shape.
 * @param y_original_shape An array of size y_original_rank representing the dimensions of the original tensor being broadcasted.
 * @param y_original_rank The order of the original tensor. Gives the number of elements in y_original_shape.
 * @param broadcasted_shape An array of size broadcasted_rank representing the dimensions of the target broadcasted tensor.
 * @param broadcasted_rank The order of the broadcasted tensor. Gives the number of elements in broadcasted_shape.
 * @return NULL if operation was successful. An error if any pointers are NULL or shapes cannot be broadcasted together.
 *         See broadcasting rules at https://numpy.org/doc/stable/user/basics.broadcasting.html.
 */
error_t *broadcast_shapes(const uint32_t *x_original_shape,
                          uint32_t x_original_rank,
                          const uint32_t *y_original_shape,
                          uint32_t y_original_rank, 
                          uint32_t *broadcasted_shape,
                          uint32_t broadcasted_rank)
{
    CHECK_NULL_ARGUMENT(x_original_shape, "x_original_shape"); 
    CHECK_NULL_ARGUMENT(y_original_shape, "y_original_shape"); 
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape"); 

    if (x_original_rank < 1 || x_original_rank > MAX_RANK || 
        y_original_rank < 1 || y_original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("x original rank %u and y original rank %u must be between 1 and %d.", 
                     (unsigned int) x_original_rank, (unsigned int) y_original_rank, (int) MAX_RANK), NULL);
    }

    for (uint32_t i = 0; i < broadcasted_rank; i++)
    {
        if ((i + 1) > x_original_rank || ((i + 1) <= y_original_rank && x_original_shape[x_original_rank - (i + 1)] == 1))
        {
            broadcasted_shape[broadcasted_rank - (i + 1)] = y_original_shape[y_original_rank - (i + 1)];
        } 
        else if ((i + 1) > y_original_rank || 
                 x_original_shape[x_original_rank - (i + 1)] == y_original_shape[y_original_rank - (i + 1)] || 
                 y_original_shape[y_original_rank - (i + 1)] == 1)
        {
            broadcasted_shape[broadcasted_rank - (i + 1)] = x_original_shape[x_original_rank - (i + 1)];
        }
        else
        {
            string_t x_original_shape_string = uint32_array_to_string(x_original_shape, x_original_rank);
            string_t y_original_shape_string = uint32_array_to_string(y_original_shape, y_original_rank);
            error_t *error = ERROR(ERROR_BROADCAST,
                                   string_create("cannot broadcast shape %s to shape %s.",
                                   x_original_shape_string,
                                   y_original_shape_string),
                                   NULL);
            string_destroy(x_original_shape_string);
            string_destroy(y_original_shape_string);
            return error;
        }
    }

    return NULL;
}

bool_t is_broadcastable(const uint32_t *original_shape,
                        uint32_t original_rank,
                        const uint32_t *broadcasted_shape,
                        uint32_t broadcasted_rank)
{
    if (original_shape == NULL || broadcasted_shape == NULL || broadcasted_rank < original_rank)
    {
        return false;
    }


    for (uint32_t i = 0; i < broadcasted_rank; i++)
    {
        if (original_rank >= (i + 1) && 
            original_shape[original_rank - (i + 1)] != broadcasted_shape[broadcasted_rank - (i + 1)] && 
            original_shape[original_rank - (i + 1)] != 1)
        {
            return false;
        }
    }

    return true;
}

error_t *reverse_broadcast_length(const uint32_t *original_shape,
                                  uint32_t original_rank,
                                  const uint32_t *broadcasted_shape,
                                  uint32_t broadcasted_rank, 
                                  uint32_t *length_keep_dimension,
                                  uint32_t *length_remove_dimension)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(length_keep_dimension, "length_keep_dimension");
    CHECK_NULL_ARGUMENT(length_remove_dimension, "length_remove_dimension");
    
    if (original_rank < 1 || original_rank > MAX_RANK || 
        broadcasted_rank < 1 || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u and broadcasted rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (unsigned int) broadcasted_rank, (int) MAX_RANK), NULL);
    }

    if (!is_broadcastable(original_shape, original_rank, broadcasted_shape, broadcasted_rank))
    {
        string_t original_shape_string = uint32_array_to_string(original_shape, original_rank);
        string_t broadcasted_shape_string = uint32_array_to_string(broadcasted_shape, broadcasted_rank);
        error_t *error = ERROR(ERROR_BROADCAST,
                               string_create("cannot broadcast shape %s to shape %s.",
                               original_shape_string,
                               broadcasted_shape_string),
                               NULL);
        string_destroy(original_shape_string);
        string_destroy(broadcasted_shape_string);
        return error;
    }

    *length_keep_dimension = 0;
    *length_remove_dimension = 0;
    for (uint32_t i = 0; i < broadcasted_rank; i++)
    {
        if (original_rank >= (i + 1))
        {
            if (original_shape[original_rank - (i + 1)] != broadcasted_shape[broadcasted_rank - (i + 1)])
            {
                (*length_keep_dimension)++;
            }
        }
        else
        {
            (*length_remove_dimension)++;
        }
    }

    return NULL;
}

error_t *reverse_broadcast_axis(const uint32_t *original_shape,
                                uint32_t original_rank,
                                const uint32_t *broadcasted_shape,
                                uint32_t broadcasted_rank, 
                                uint32_t *axis_keep_dimension,
                                uint32_t *axis_remove_dimension)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(axis_keep_dimension, "axis_keep_dimension");
    CHECK_NULL_ARGUMENT(axis_remove_dimension, "axis_remove_dimension");

    if (original_rank < 1 || original_rank > MAX_RANK || 
        broadcasted_rank < 1 || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u and broadcasted rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (unsigned int) broadcasted_rank, (int) MAX_RANK), NULL);
    }

    if (!is_broadcastable(original_shape, original_rank, broadcasted_shape, broadcasted_rank))
    {
        string_t original_shape_string = uint32_array_to_string(original_shape, original_rank);
        string_t broadcasted_shape_string = uint32_array_to_string(broadcasted_shape, broadcasted_rank);
        error_t *error = ERROR(ERROR_BROADCAST,
                               string_create("cannot broadcast shape %s to shape %s.",
                               original_shape_string,
                               broadcasted_shape_string),
                               NULL);
        string_destroy(original_shape_string);
        string_destroy(broadcasted_shape_string);
        return error;
    }

    uint32_t j = 0;
    uint32_t k = 0;
    for (uint32_t i = 0; i < broadcasted_rank; i++)
    {
        if (original_rank >= (i + 1))
        {
            if (original_shape[original_rank - (i + 1)] != broadcasted_shape[broadcasted_rank - (i + 1)])
            {
                axis_keep_dimension[j] = broadcasted_rank - (i + 1);
                j++;
            }
        }
        else
        {
            axis_remove_dimension[k] = broadcasted_rank - (i + 1);
            k++;
        }
    }

    return NULL;
}

error_t *slice_shape(const uint32_t *original_shape,
                     uint32_t original_rank,
                     uint32_t *slice_shape,
                     uint32_t slice_rank,
                     const uint32_t *arguments,
                     uint32_t length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(slice_shape, "slice_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    if (original_rank != slice_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with original rank %u and sliced rank %u.", 
                     (unsigned int) original_rank, (unsigned int) slice_rank), NULL);
    }

    if (original_rank < 1 || original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (int) MAX_RANK), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with original rank %u and axis length %u which should be a multiple of 2.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %u and length of arguments %u.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    for (uint32_t i = 0; i < original_rank; i += 2)
    {
        if (arguments[2 * i + 1] <= arguments[2 * i] ||
            arguments[2 * i] > original_shape[i] ||
            arguments[2 * i + 1] > original_shape[i])
        {
            return ERROR(ERROR_SHAPE_CONFLICT, 
                         string_create("upperbound of slice %u must be greater than lower bound %u and bounds must be less than dimension %u.", 
                         (unsigned int) arguments[2 * i + 1], (unsigned int) arguments[2 * i], (unsigned int) original_shape[i]), NULL);
        }
        slice_shape[i] = (arguments[2 * i] - arguments[2 * i + 1]); 
    }

    return NULL;
}

error_t *slice_offset(const uint32_t *original_strides,
                      uint32_t original_rank,
                      uint32_t *offset,
                      const uint32_t *arguments,
                      uint32_t length)
{
    CHECK_NULL_ARGUMENT(original_strides, "original_strides");
    CHECK_NULL_ARGUMENT(offset, "offset");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    if (original_rank < 1 || original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (int) MAX_RANK), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with original rank %u and axis length %u which should be a multiple of 2.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %u and length of arguments %u.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    *offset = 0;
    for (uint32_t i = 0; i < length; i += 2)
    {
        *offset += original_strides[i / 2] * arguments[i];
    }

    return NULL;
}

error_t *reverse_slice(const uint32_t *original_shape,
                       uint32_t original_rank,
                       const uint32_t *arguments,
                       uint32_t length,
                       uint32_t *new_arguments,
                       uint32_t new_length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(new_arguments, "new_arguments");

    if (new_length != length)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %u is not equal to length of new arguments %u.", 
                     (unsigned int) length, (unsigned int) new_length), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %u is not a multiple of 2.", 
                     (unsigned int) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %u and length of arguments %u.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    if (original_rank < 1 || original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (int) MAX_RANK), NULL);
    }

    for (uint32_t i = 0; i < new_length; i += 2)
    {
        new_arguments[i] = arguments[i];
        new_arguments[i + 1] = original_shape[i / 2] - arguments[i + 1];
    }

    return NULL;
}

error_t *padding(const uint32_t *original_shape,
                 uint32_t original_rank,
                 uint32_t *padding_shape,
                 uint32_t padding_rank,
                 const uint32_t *arguments,
                 uint32_t length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(padding_shape, "padding_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    if (original_rank != padding_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with original rank %u, padding rank %u.", 
                     (unsigned int) original_rank, (unsigned int) padding_rank), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of arguments %u is not a multiple of 2.", 
                     (unsigned int) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %u and length of arguments %u.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    if (original_rank < 1 || original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (int) MAX_RANK), NULL);
    }

    for (uint32_t i = 0; i < original_rank; i++)
    {
        if (i < length / 2)
        {
            padding_shape[i] += arguments[2 * i] + arguments[2 * i + 1]; 
        }
        else
        {
            padding_shape[i] = original_shape[i];
        }
    }

    return NULL;
}

error_t *reverse_padding(const uint32_t *original_shape, 
                         uint32_t original_rank,
                         const uint32_t *arguments,
                         uint32_t length,
                         uint32_t *new_arguments,
                         uint32_t new_length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(new_arguments, "new_arguments");

    if (new_length != length)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %u is not equal to length of new arguments %u.", 
                     (unsigned int) length, (unsigned int) new_length), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %u is not a multiple of 2.", 
                     (unsigned int) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %u and length of arguments %u.",
                     (unsigned int) original_rank, (unsigned int) length), NULL);
    }

    if (original_rank < 1 || original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %u must be between 1 and %d.", 
                     (unsigned int) original_rank, (int) MAX_RANK), NULL);
    }

    for (uint32_t i = 0; i < new_length; i += 2)
    {
        new_arguments[i] = arguments[i];
        new_arguments[i + 1] = original_shape[i / 2] + arguments[i + 1];
    }

    return NULL;
}