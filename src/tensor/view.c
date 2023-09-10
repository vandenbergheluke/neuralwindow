/**
 * @file view.c
 * @brief The view defines an interpretation of the underlying storage 
 *        used to represent a tensor. File contains operations to create, 
 *        manipulate, and describe the view.
 */

#include <view.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Dynamically memory allocate and initialize a view. 
 *        Refernence: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html       
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
 * @return Error if failed to dynamically allocate memory for view or any of
 *         its members.
 *         Error if `rank` does not satisfy `0 <= rank <= MAX_RANK`.
 *         Error of the contiguous tensor strides failed to be computed.
 *         Error if a dimension in shape is 0.
 *         NULL if view was successfully dynamically memory allocated and initialized.
 */
nw_error_t *view_create(view_t **view,
                        uint64_t offset,
                        uint64_t rank,
                        const uint64_t *shape,
                        const uint64_t *strides)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(shape, "shape");

    if (rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank %lu must be less or equal to %d.",
                     (unsigned long) rank, (int) MAX_RANK), NULL);
    }

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (!shape[i])
        {
            return ERROR(ERROR_SHAPE_CONFLICT,
                         string_create("all tensor dimensions must be greater than 0."),
                         NULL);
        }
    }

    // View
    *view = (view_t *) malloc((size_t) sizeof(view_t));
    if (view == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view of size %lu.",
                     (unsigned long) sizeof(view_t)), NULL);
    } 

    // Initialize
    (*view)->offset = offset;
    (*view)->rank = rank;

    // Dynamically allocate memory for shape

    // If rank is 0, this should return a reference to a block of memory of size 0.
    (*view)->shape = (uint64_t *) malloc((size_t) (rank * sizeof(uint64_t)));
    if ((*view)->shape == NULL)
    {
        free(*view);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view->shape of size %lu.",
                     (unsigned long) (rank * sizeof(uint64_t))), NULL);
    }

    // Dynamically allocate memory for strides.

    // If rank is 0, this should return a reference to a block of memory of size 0.
    (*view)->strides = (uint64_t *) malloc((size_t) (rank * sizeof(uint64_t)));
    if ((*view)->strides == NULL)
    {
        free(*view);
        free((*view)->shape);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view->strides of size %lu.",
                     (unsigned long) (rank * sizeof(uint64_t))), NULL);
    }

    // Copy

    // Don't need to copy for scalar tensors.
    if (rank == 0)
    {
       return NULL; 
    }

    // Initialize Shape
    memcpy((void *) ((*view)->shape), 
           (const void *) shape, (size_t) (rank * sizeof(uint64_t)));

    // Initialize Strides
    if (strides != NULL)
    {
        memcpy((void *) ((*view)->strides), 
               (const void *) strides,
               (size_t) (rank * sizeof(uint64_t)));
    }
    else
    {
        nw_error_t *error = strides_from_shape((*view)->strides, shape, rank);
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

    return NULL;
}

/**
 * @brief Free memory dynamically allocated for view and its members.
 * @param[in] view The view instance to be freed.
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
 *        Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html
 *        Reference: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py 
 * @param[in] shape The dimensions of the tenors. 
 * @param[in] rank  Represents rank of the tensor. The number of elements in
 *                  `shape` and `strides`. Needs to be in closed interval `[0, MAX_RANK]`.
 * @param[in] strides The strides are the jumps necessary to go from one element 
 *                    to the next one in storage along each dimension.
 * @param[in] offset The offset is the number of elements to skip in the memory block
 *                   to arrive at the first element of the tensor.
 * @return True if the tensor memory is contiguous and False if it isn't.
 *         If any argument is NULL, or `rank` does not satisfy `0 <= rank <= MAX_RANK`
 *         false is returned. If error occured while computing contiguous tensor
 *         strides false is also returned.
 */
bool_t is_contiguous(const uint64_t *shape, uint64_t rank, const uint64_t *strides, uint64_t offset)
{
    if (shape == NULL || strides == NULL)
    {
        return false;
    }

    if (rank > MAX_RANK)
    {
        return false;
    }

    if (offset)
    {
        return false;
    }

    uint64_t contiguous_strides[rank];    
    nw_error_t *error = strides_from_shape(contiguous_strides, shape, rank);
    if (error != NULL)
    {
        error_destroy(error);
        return false;
    }

    for (uint64_t i = 0; i < rank; ++i)
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
 *        Refernece: https://pytorch.org/docs/stable/generated/torch.permute.html
 * @param[in] original_shape The original dimensions of the tensor. 
 * @param[in] original_strides The original memory strides.
 * @param[out] permuted_shape The reordered dimensions.
 * @param[out] permuted_strides The reordered memory strides.
 * @param[in] axis Index array containing the order of dimensions.
 * @param[in] length Number of elements `original_shape`, `original_strides`,
 *                   `permuted_shape`, `permuted_strides`, and `axis`. Should
 *                   be in closed interval [0, MAX_RANK].
 * @return Error if length does not satisfy 0 <= length <= MAX_RANK.
 *         Error if any argument is NULL.
 *         Error if dimension index in `axis` is greater than length (out of range).
 *         Error if not all indicies in `axis` are unique.
 *         NULL if dimensions of the tensor were successfully permuted to the 
 *         new order specified by axis.
 */
nw_error_t *permute(const uint64_t *original_shape,
                    const uint64_t *original_strides, 
                    uint64_t *permuted_shape,
                    uint64_t *permuted_strides,
                    const uint64_t *axis,
                    uint64_t length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides, "original_strides");
    CHECK_NULL_ARGUMENT(permuted_shape, "permuted_shape");
    CHECK_NULL_ARGUMENT(permuted_strides, "permuted_strides");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_UNIQUE(axis, length, "axis");

    if (length > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("axis length %lu must be less than or equal to %d.",
                     (unsigned long) length, (int) MAX_RANK),
                     NULL);
    }
    
    for (uint64_t i = 0; i < length; ++i)
    {
        uint64_t dimension = axis[i];
        if (dimension < length)
        {
            if (!original_shape[dimension])
            {
                return ERROR(ERROR_SHAPE_CONFLICT,
                             string_create("all shape dimensions must be greater than 0."),
                             NULL);
            }

            permuted_shape[i] = original_shape[dimension];
            permuted_strides[i] = original_strides[dimension];
        }
        else
        {
            return ERROR(ERROR_PERMUTE,
                         string_create("axis dimension %lu out of range of length %lu.", 
                         (unsigned long) dimension, (unsigned long) length), NULL);
        }
    }
    
    return NULL;
}

typedef struct pair_t
{
    uint64_t index;
    uint64_t value;
} pair_t;
    
static int compare(const void *a, const void *b)
{
    pair_t *pair_a = (pair_t *) a;
    pair_t *pair_b = (pair_t *) b;

    return (pair_a->value - pair_b->value);
}

/**
 * @brief Given an `axis` of size `rank` used to reorder the dimensions of a 
 *        tensor through the `permute` operation, find a new axis `reverse_axis`
 *        such that when the `permute` operation with `reverse_axis` is applied 
 *        to a tensor that has been permuted with `axis`, the original dimension
 *        order of the tensor before being permuted with `axis` is recovered. 
 *        Equivalently, this is an argsort of axis. 
 *        Reference: https://pytorch.org/docs/stable/generated/torch.argsort.html 
 * @param[in] axis An array of indicies specifying the order of dimensions.
 * @param[in] rank Number of elements in `axis` and `reverse_axis` that is equal to
 *                 the rank of the permuted tensor.
 * @param[out] reverse_axis An array of indicies specifying the order of dimensions
 *                          to reverse the permutation operation that was applied
 *                          with `axis`.
 * @return Error if `axis` or `reverse_axis` is NULL.
 *         Error if memory has failed to be dynamically allocated for axis.
 *         Error if `rank` does not satisfy 0 <= rank <= 5.
 *         Error if `axis` dimension index is greater than or equal to rank.
 *         Error if not all dimension indices in `axis` are unique.
 *         NULL if `reverse_axis` was successfully acquired.
 */
nw_error_t *reverse_permute(const uint64_t *axis, uint64_t rank, uint64_t *reverse_axis)
{
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(reverse_axis, "reverse_axis");
    CHECK_UNIQUE(axis, rank, "axis");

    if (rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank %lu must be less than or equal to %d.",
                     (unsigned long) rank, (int) MAX_RANK), NULL);
    }

    pair_t *new_axis = (pair_t *) malloc((size_t) (rank * sizeof(pair_t)));
    if (new_axis == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate new axis of size %lu bytes.",
                     (unsigned long) (rank * sizeof(pair_t))), NULL);
    }

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (axis[i] >= rank)
        {
            free(new_axis);
            return ERROR(ERROR_RANK_CONFLICT,
                         string_create("dimension index %lu cannot be greater or equal to rank %lu",
                         (unsigned long) axis[i], (unsigned long) rank),
                         NULL);
        }

        new_axis[i].index = i;
        new_axis[i].value = axis[i];
    }

    qsort((void *) new_axis, (size_t) rank, sizeof(pair_t), compare);

    for (uint64_t i = 0; i < rank; ++i)
    {
        reverse_axis[i] = new_axis[i].index;
    }

    free(new_axis);

    return NULL;
}

/**
 * @brief Given a the shape and strides of a tensor that has been reduced without keeping
 *        the dimension, recover the shape and strides of the reduced tensor such that
 *        the recovered shape and strides are equal to the shape and strides of the
 *        tensor if it was reduced with keeping the dimension instead of without keeping dimension.
 * @param[in] reduced_shape The shape of a tensor reduced along dimensions specified by `axis`
 *                          without keeping dimension.
 * @param[in] reduced_rank The rank of the reduced tensor. Number of elements in `reduced_shape`
 *                         and `reduced_strides`.
 * @param[in] reduced_strides The strides of the reduced tensor.
 * @param[out] recovered_shape The shape of the reduced tensor if it was reduced with keeping dimension
 *                             instead of without keeping dimension.
 * @param[in] recovered_rank The rank of the original tensor before being reduced. Number of
 *                           elements in `recovered_shape` and `recovered_strides`.
 * @param[out] recovered_strides The strides of the recovered tensor.
 * @param[in] axis The indicies of the dimensions of the tensor that were reduced and removed.
 * @param[in] rank The number of elements in `axis`.
 * @return Error if `reduced_shape`, `reduced_strides`, `recovered_shape`, `recovered_strides`, or `axis` is NULL.
 *         Error if `reduced_rank` + `rank` is not equal to the `recovered_rank`.
 *         Error if any of the `reduced_rank`, `recovered_rank`, or `rank` are not in [0, MAX_RANK].
 *         Error if `axis` dimension index is out of range of the `recovered_rank`.
 *         Error if any dimension in `reduced_shape` is 0.
 *         Error if not all dimension indices in `axis` are unique.
 *         NULL if the shape and strides were recovered successfully.
 */
nw_error_t *reduce_recover_dimensions(const uint64_t *reduced_shape,
                                      uint64_t reduced_rank, 
                                      const uint64_t *reduced_strides,
                                      uint64_t *recovered_shape, 
                                      uint64_t recovered_rank,
                                      uint64_t *recovered_strides,
                                      const uint64_t *axis,
                                      uint64_t rank)
{
    CHECK_NULL_ARGUMENT(reduced_shape, "reduced_shape");
    CHECK_NULL_ARGUMENT(reduced_strides, "reduced_strides");
    CHECK_NULL_ARGUMENT(recovered_shape, "recovered_shape");
    CHECK_NULL_ARGUMENT(recovered_strides , "recovered_strides");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_UNIQUE(axis, rank, "axis");

    if (recovered_rank != reduced_rank + rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with reduced rank %lu, recovered rank %lu and axis length %lu.",
                     (unsigned long) reduced_rank, (unsigned long) recovered_rank, (unsigned long) rank),
                     NULL);
    }

    if (reduced_rank > MAX_RANK || recovered_rank > MAX_RANK || rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("reduced rank %lu, recovered rank %lu and axis length %lu must be less than or equal to %d.",
                     (unsigned long) reduced_rank, (unsigned long) recovered_rank, (unsigned long) rank, (int) MAX_RANK),
                     NULL);
    }

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (axis[i] >= recovered_rank)
        {
            return ERROR(ERROR_RANK_CONFLICT,
                         string_create("recovered rank %lu must be greater than the axis dimension index %lu.",
                         (unsigned long) recovered_rank, (unsigned long) axis[i]),
                         NULL);
        }
    }

    uint64_t k = 0;
    for (uint64_t i = 0; i < recovered_rank; ++i)
    {
        bool_t reduced = false;
        for (uint64_t j = 0; j < rank; ++j)
        {
            if (axis[j] == i)
            {
                recovered_shape[i] = 1;
                recovered_strides[i] = 0;
                reduced = true;
                break;
            }
        }
    
        if (!reduced)
        {
            if (k >= reduced_rank)
            {
                return ERROR(ERROR_RANK_CONFLICT,
                             string_create("error index %lu out of range of reduced rank %lu.",
                             (unsigned long) k, (unsigned long) reduced_rank),
                             NULL);
            }

            if (!reduced_shape[k])
            {
                return ERROR(ERROR_SHAPE_CONFLICT,
                             string_create("all reduced shape dimensions must be greater than 0."),
                             NULL);
            }

            recovered_shape[i] = reduced_shape[k];
            recovered_strides[i] = reduced_strides[k];
            ++k;
        }
    }

    return NULL;
}

/**
 * @brief Given the shape, rank, and strides of a tensor, find the resulting
 *        shape and strides of the tensor after it is reduced.
 * @param[in] original_shape The original dimensions of the tensor before it is reduced.
 * @param[in] original_rank The original rank of the tensor before it is reduced. 
 *                          The number of elements in `original_shape` and `original_strides`.
 * @param[in] original_strides The original strides of the tensor before it is reduced.
 * @param[out] reduced_shape The dimensions of the tensor after it is reduced.
 * @param[in] reduced_rank The rank of the tensor after it is reduced. 
 *                         The number of elements in `reduced_shape` and `reduced_strides`.
 * @param[out] reduced_strides The strides of the tensor after it is reduced.
 * @param[in] axis An array of inidices of tensor dimensions to reduce.
 * @param[in] rank The number of indices in `axis`.
 * @param[in] keep_dimensions A flag to indicate whether the dimensions are retained (true)
 *                            after reduction or removed (false).
 * @return Error if `original_shape`, `original_strides`, `reduced_shape`, `reduced_strides`,
 *         or `axis` are NULL.
 *         Error if `reduced_shape` or `reduced_strides` failed to be computed.
 *         Error if not all dimension indices in `axis` are unique.
 *         NULL if reduced shape and strides were successfully computed.
 */
nw_error_t *reduce(const uint64_t *original_shape,
                   uint64_t original_rank,
                   const uint64_t *original_strides, 
                   uint64_t *reduced_shape,
                   uint64_t reduced_rank,
                   uint64_t *reduced_strides,
                   const uint64_t *axis,
                   uint64_t rank,
                   bool_t keep_dimensions)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides , "original_strides");
    CHECK_NULL_ARGUMENT(reduced_shape, "reduced_shape");
    CHECK_NULL_ARGUMENT(reduced_strides, "reduced_strides");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_UNIQUE(axis, rank, "axis");

    if (rank > original_rank || original_rank > MAX_RANK || reduced_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("original rank %lu, reduced rank %lu and axis length %lu must be less than or equal to %d and rank <= original rank.",
                     (unsigned long) original_rank, (unsigned long) reduced_rank, (unsigned long) rank, (int) MAX_RANK),
                     NULL);
    }

    if (keep_dimensions && original_rank != reduced_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with original rank %lu and reduced rank %lu.",
                     (unsigned long) original_rank, (unsigned long) reduced_rank),
                     NULL);
    }

    if (!keep_dimensions && reduced_rank != original_rank - rank)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with expected reduced rank %lu and reduced rank %lu.", 
                     (unsigned long) (original_rank - rank), (unsigned long) reduced_rank),
                     NULL);
    }


    for (uint64_t i = 0; i < rank; ++i)
    {
        if (axis[i] >= original_rank)
        {
            return ERROR(ERROR_RANK_CONFLICT,
                         string_create("original rank %lu must be greater than axis dimension index %lu.",
                         (unsigned long) original_rank, (unsigned long) axis[i]),
                         NULL);
        }
    }

    if (!reduced_rank || !original_rank)
    {
        return NULL;
    }

    uint64_t k = reduced_rank - 1;
    uint64_t stride = 1;

    for (uint64_t i = original_rank - 1; ; i--)
    {
        bool_t reduce_dimension = false;
        for (uint64_t j = 0; j < rank; ++j)
        {
            if (axis[j] == i)
            {
                reduce_dimension = true;
                break;
            }
        }

        if (original_shape[i] == 0)
        {
            return ERROR(ERROR_SHAPE_CONFLICT,
                         string_create("The dimensions of the original shape must all be greater than 0"),
                         NULL);
        }

        if (reduce_dimension && keep_dimensions)
        {
            reduced_shape[k] = 1;
            reduced_strides[k] = 0;
            k--;
        }
        else if (!reduce_dimension)
        {
            reduced_shape[k] = original_shape[i];
            if (original_strides[i] == 0)
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

        if (i == 0)
        {
            break;
        }
    }

    return NULL;
}

/**
 * @brief Given the shape and strides of a tensor, compute the required
 *        number of tensor data elements that need to be stored in memory.
 * @param[in] shape The dimensions of the tensor.
 * @param[in] strides The strides of the tensor.
 * @param[in] rank The number of dimensions in `shape`.
 * @param[out] n The number of elements that are required to be stored in 
 *               memory to represent the tensor.
 * @return Error if `shape`, `strides`, or `n` is NULL.
 *         Error if any of the dimensions is zero.
 *         Error if `rank` greater than max rank.
 *         NULL if `n` is computed successfully.
 */
nw_error_t *n_from_shape_and_strides(const uint64_t *shape, 
                                     const uint64_t *strides,
                                     uint64_t rank,
                                     uint64_t *n)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(strides, "strides");
    CHECK_NULL_ARGUMENT(n, "n");

    if (rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank %lu must be less than or equal to %d.",
                     (unsigned long) rank, (int) MAX_RANK),
                     NULL);
    }

    *n = shape_size(shape, rank);
    if (!(*n))
    {
        ++(*n);
    }

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (!strides[i])
        {
            if (shape[i])
            {
                *n /= shape[i];
            }
            else
            {
                return ERROR(ERROR_SHAPE_CONFLICT,
                             string_create("all dimensions of the tensor must be greater than 0."),
                             NULL);
            }
        }
    }

    return NULL;
}


/**
 * @brief Given the shape and rank of two tensors, determine if both tensors have same rank and dimensions. 
 * @param[in] x_shape An array of size `x_rank` representing the dimensions of a tensor.
 * @param[in] x_rank The order of the tensor. Gives number of elements in `x_shape`.
 * @param[in] y_shape An array of size `y_rank` representing the dimensions of a tensor.
 * @param[in] y_rank The order of the tensor. Gives number of elements in `y_shape`.
 * @return False if either shapes are NULL, ranks are not equal, or shape dimensions at a common index are not equal. 
 *         True if ranks are equal and shape dimensions are equal.
 */
bool_t shapes_equal(const uint64_t *x_shape, uint64_t x_rank, const uint64_t *y_shape, uint64_t y_rank)
{
    if (x_shape == NULL || y_shape == NULL || x_rank != y_rank)
    {
        return false;
    }

    for (uint64_t i = 0; i < x_rank; ++i)
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
 *        Note that this is just the percieved number of elements in the tensor and 
 *        may not be equal to the number of elements actually stored in memory.
 *        The actual number of elements stored in memory is defined by `n` in `buffer`. 
 *        Reference: https://pytorch.org/docs/stable/generated/torch.numel.html
 * @param[in] shape An array of size rank representing the dimensions of a tensor.
 * @param[in] rank The order of the tensor. Gives number of elements in shape.
 * @return The percieved number of elements in the tensor.
 *         Returns 0 if shape is NULL.
 */
uint64_t shape_size(const uint64_t *shape, uint64_t rank)
{
    uint64_t total = 1;

    if (shape == NULL)
    {
        return total;
    }

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (!shape[i])
        {
            continue;
        }

        total *= shape[i];
    }
    
    return total;
}

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
nw_error_t *strides_from_shape(uint64_t *strides, const uint64_t *shape, uint64_t rank)
{
    CHECK_NULL_ARGUMENT(strides, "strides");
    CHECK_NULL_ARGUMENT(shape, "shape");

    if (rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("rank %lu must be less than or equal to %d.", 
                     (unsigned long) rank, (int) MAX_RANK),
                     NULL);
    }

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (!shape[i])
        {
            return ERROR(ERROR_SHAPE_CONFLICT,
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

    for (uint64_t i = 0; i < rank; ++i)
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
 * @param[in] original_shape An array of size original_rank representing the dimensions of the original tensor being broadcasted.
 * @param[in] original_rank The order of the original tensor. Gives the number of elements in original_shape.
 * @param[in] original_strides The memory strides to traverse elements of the original tensor.
 * @param[in] broadcasted_shape An array of size broadcasted_rank representing the dimensions of the target broadcasted tensor.
 * @param[in] broadcasted_rank The order of the broadcasted tensor. Gives the number of elements in broadcasted_shape.
 * @param[out] broadcasted_strides The memory strides to traverse elements of the broadcasted tensor.
 * @return NULL if operation was successful. An error if any pointers are NULL or shapes cannot be broadcasted together.
 *         See broadcasting rules at https://numpy.org/doc/stable/user/basics.broadcasting.html.
 */
nw_error_t *broadcast_strides(const uint64_t *original_shape,
                              uint64_t original_rank,
                              const uint64_t *original_strides,
                              const uint64_t *broadcasted_shape,
                              uint64_t broadcasted_rank,
                              uint64_t *broadcasted_strides)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(original_strides, "original_strides");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(broadcasted_strides, "broadcasted_strides");

    if (original_rank > MAX_RANK || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu and broadcasted rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (unsigned long) broadcasted_rank, (int) MAX_RANK),
                     NULL);
    }

    if (!is_broadcastable(original_shape, original_rank, broadcasted_shape, broadcasted_rank))
    {
        return ERROR(ERROR_BROADCAST,
                     string_create("cannot broadcast shapes."),
                     NULL);
    }

    for (uint64_t i = 1; i < broadcasted_rank + 1; ++i)
    {   
        uint64_t original_index = original_rank - i;
        uint64_t broadcast_index = broadcasted_rank - i;


        if (i > original_rank || (original_shape[original_index] == 1))
        {
            broadcasted_strides[broadcast_index] = 0; 
        }
        else if (original_shape[original_index] == broadcasted_shape[broadcast_index])
        {
            if (!original_shape[original_index] || !broadcasted_shape[broadcast_index])
            {
                return ERROR(ERROR_SHAPE_CONFLICT,
                            string_create("all shape dimensions must be greater than 0."),
                            NULL);
            }

            broadcasted_strides[broadcast_index] = original_strides[original_index];
        }
        else
        {
            return ERROR(ERROR_BROADCAST,
                         string_create("cannot broadcast shape."),
                         NULL);
        }
    }

    return NULL;
}

/**
 * @brief Given the shape, rank, and strides of two tensors being combined via an elementwise binary operation, find
 *        the associated shape and rank to broadcast both tensors to perform the operation.   
 * @param[in] x_original_shape An array of size `x_original_rank` representing the dimensions of the original tensor being broadcasted.
 * @param[in] x_original_rank The order of the original tensor. Gives the number of elements in `x_original_shape`.
 * @param[in] y_original_shape An array of size `y_original_rank` representing the dimensions of the original tensor being broadcasted.
 * @param[in] y_original_rank The order of the original tensor. Gives the number of elements in `y_original_shape`.
 * @param[out] broadcasted_shape An array of size `broadcasted_rank` representing the dimensions of the target broadcasted tensor.
 * @param[in] broadcasted_rank The order of the broadcasted tensor. Gives the number of elements in `broadcasted_shape`.
 * @return NULL if operation was successful. An error if any pointers are NULL or shapes cannot be broadcasted together.
 *         See broadcasting rules at https://numpy.org/doc/stable/user/basics.broadcasting.html.
 */
nw_error_t *broadcast_shapes(const uint64_t *x_original_shape,
                             uint64_t x_original_rank,
                             const uint64_t *y_original_shape,
                             uint64_t y_original_rank, 
                             uint64_t *broadcasted_shape,
                             uint64_t broadcasted_rank)
{
    CHECK_NULL_ARGUMENT(x_original_shape, "x_original_shape"); 
    CHECK_NULL_ARGUMENT(y_original_shape, "y_original_shape"); 
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape"); 

    if (x_original_rank > MAX_RANK || y_original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("x original rank %lu and y original rank %lu must be less than or equal to %d.", 
                     (unsigned long) x_original_rank, (unsigned long) y_original_rank, (int) MAX_RANK),
                     NULL);
    }

    if (broadcasted_rank != MAX(x_original_rank, y_original_rank))
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("broadcast rank %lu must be the max rank of {%lu, %lu}.",
                     (unsigned long) broadcasted_rank, (unsigned long) x_original_rank, (unsigned long) y_original_rank),
                     NULL);
    }

    for (uint64_t i = 1; i < broadcasted_rank + 1; ++i)
    {
        uint64_t x_index = x_original_rank - i;
        uint64_t y_index = y_original_rank - i;
        uint64_t broadcast_index = broadcasted_rank - i;
        if (i > x_original_rank || (i <= y_original_rank && x_original_shape[x_index] == 1))
        {
            broadcasted_shape[broadcast_index] = y_original_shape[y_index];
        } 
        else if (i > y_original_rank || 
                 x_original_shape[x_index] == y_original_shape[y_index] || 
                 y_original_shape[y_index] == 1)
        {
            broadcasted_shape[broadcast_index] = x_original_shape[x_index];
        }
        else
        {
            return ERROR(ERROR_BROADCAST,
                        string_create("cannot broadcast shapes."),
                        NULL);
        }
    }

    return NULL;
}

/**
 * @brief Given the shape, rank, and strides of two tensors being combined via an matrix multiplication operation, find
 *        the associated shape and rank to broadcast both tensors to perform the operation.   
 * @param[in] x_original_shape An array of size `x_original_rank` representing the dimensions of the original tensor being broadcasted.
 * @param[in] x_original_rank The order of the original tensor. Gives the number of elements in `x_original_shape`.
 * @param[in] y_original_shape An array of size `y_original_rank` representing the dimensions of the original tensor being broadcasted.
 * @param[in] y_original_rank The order of the original tensor. Gives the number of elements in `y_original_shape`.
 * @param[out] x_broadcasted_shape An array of size `broadcasted_rank` representing the dimensions of the target broadcasted tensor of `x`.
 * @param[out] y_broadcasted_shape An array of size `broadcasted_rank` representing the dimensions of the target broadcasted tensor of `y`.
 * @param[in] broadcasted_rank The order of the broadcasted tensor. Gives the number of elements in `broadcasted_shape`.
 * @return NULL if operation was successful. An error if any pointers are NULL or shapes cannot be broadcasted together.
 *         See broadcasting rules at https://pytorch.org/docs/stable/generated/torch.matmul.html.
 */
nw_error_t *matrix_multiplication_broadcast_shapes(const uint64_t *x_original_shape,
                                                   uint64_t x_original_rank,
                                                   const uint64_t *y_original_shape,
                                                   uint64_t y_original_rank, 
                                                   uint64_t *x_broadcasted_shape,
                                                   uint64_t *y_broadcasted_shape,
                                                   uint64_t broadcasted_rank)
{
    CHECK_NULL_ARGUMENT(x_original_shape, "x_original_shape"); 
    CHECK_NULL_ARGUMENT(y_original_shape, "y_original_shape"); 
    CHECK_NULL_ARGUMENT(x_broadcasted_shape, "x_broadcasted_shape"); 
    CHECK_NULL_ARGUMENT(y_broadcasted_shape, "y_broadcasted_shape"); 

    if (x_original_rank > MAX_RANK ||
        y_original_rank > MAX_RANK ||
        x_original_rank < 2 ||
        y_original_rank < 2)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("x original rank %lu and y original rank %lu must be in the interval [2, %d].", 
                     (unsigned long) x_original_rank, (unsigned long) y_original_rank, (int) MAX_RANK),
                     NULL);
    }

    if (broadcasted_rank != MAX(x_original_rank, y_original_rank))
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("broadcast rank %lu must be the max rank of {%lu, %lu}.",
                     (unsigned long) broadcasted_rank, (unsigned long) x_original_rank, (unsigned long) y_original_rank),
                     NULL);
    }

    for (uint64_t i = 1; i < broadcasted_rank + 1; ++i)
    {
        uint64_t x_index = x_original_rank - i;
        uint64_t y_index = y_original_rank - i;
        uint64_t broadcast_index = broadcasted_rank - i;
        if (i < 3)
        {
            x_broadcasted_shape[broadcast_index] = x_original_shape[x_index];
            y_broadcasted_shape[broadcast_index] = y_original_shape[y_index];
            continue;
        }

        if (i > x_original_rank || (i <= y_original_rank && x_original_shape[x_index] == 1))
        {
            x_broadcasted_shape[broadcast_index] = y_original_shape[y_index];
            y_broadcasted_shape[broadcast_index] = y_original_shape[y_index];
        } 
        else if (i > y_original_rank || 
                 x_original_shape[x_index] == y_original_shape[y_index] || 
                 y_original_shape[y_index] == 1)
        {
            x_broadcasted_shape[broadcast_index] = x_original_shape[x_index];
            y_broadcasted_shape[broadcast_index] = x_original_shape[x_index];
        }
        else
        {
            return ERROR(ERROR_BROADCAST,
                        string_create("cannot broadcast shapes."),
                        NULL);
        }
    }

    return NULL;
}

nw_error_t *matrix_multiplication_shape(uint64_t *x_shape, uint64_t *y_shape, uint64_t *z_shape, uint64_t rank)
{
    CHECK_NULL_ARGUMENT(x_shape, "x_shape");
    CHECK_NULL_ARGUMENT(y_shape, "y_shape");
    CHECK_NULL_ARGUMENT(z_shape, "z_shape");

    if (rank < 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank %lu must be 2 or greater.",
                     (unsigned long) rank),
                     NULL);
    }

    if (x_shape[rank - 1] != y_shape[rank - 2])
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("number of columns in x %lu not equal to number of rows in y %lu.",
                     (unsigned long) x_shape[rank - 1], (unsigned long) y_shape[rank - 1]),
                     NULL);
    }

    for (uint64_t i = 1; i < rank + 1; ++i)
    {
        uint64_t j = rank - i;
        if (i == 1)
        {
            z_shape[j] = y_shape[j];
        }
        else if (i == 2)
        {
            z_shape[j] = x_shape[j];
        }
        else
        {
            if (x_shape[j] != y_shape[j])
            {
                return ERROR(ERROR_SHAPE_CONFLICT,
                             string_create("dimension in x %lu not equal to dimension in y $lu.",
                             (unsigned long) x_shape[j], (unsigned long) y_shape[j]),
                             NULL);
            }
            else
            {
                z_shape[j] = x_shape[j];
            }
        }
    }

    return NULL;
}

/**
 * @brief 
 * 
 * @param original_shape 
 * @param original_rank 
 * @param broadcasted_shape 
 * @param broadcasted_rank 
 * @return bool_t 
 */
bool_t is_broadcastable(const uint64_t *original_shape,
                        uint64_t original_rank,
                        const uint64_t *broadcasted_shape,
                        uint64_t broadcasted_rank)
{
    if (original_shape == NULL ||
        broadcasted_shape == NULL ||
        broadcasted_rank < original_rank)
    {
        return false;
    }


    for (uint64_t i = 1; i < broadcasted_rank + 1; ++i)
    {
        if (original_rank >= i && 
            original_shape[original_rank - i] != broadcasted_shape[broadcasted_rank - i] && 
            original_shape[original_rank - i] != 1)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief 
 * 
 * @param original_shape 
 * @param original_rank 
 * @param broadcasted_shape 
 * @param broadcasted_rank 
 * @param length_keep_dimension 
 * @param length_remove_dimension 
 * @return 
 */
nw_error_t *reduce_axis_length(const uint64_t *original_shape,
                               uint64_t original_rank,
                               const uint64_t *broadcasted_shape,
                               uint64_t broadcasted_rank, 
                               uint64_t *length_keep_dimension,
                               uint64_t *length_remove_dimension)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(length_keep_dimension, "length_keep_dimension");
    CHECK_NULL_ARGUMENT(length_remove_dimension, "length_remove_dimension");
    
    if (original_rank > MAX_RANK || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu and broadcasted rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (unsigned long) broadcasted_rank, (int) MAX_RANK),
                     NULL);
    }

    if (!is_broadcastable(original_shape, original_rank, broadcasted_shape, broadcasted_rank))
    {
        return ERROR(ERROR_BROADCAST,
                     string_create("cannot broadcast shapes."),
                     NULL);
    }

    *length_keep_dimension = 0;
    *length_remove_dimension = 0;
    for (uint64_t i = 0; i < broadcasted_rank; ++i)
    {
        if (original_rank >= (i + 1))
        {
            if (original_shape[original_rank - (i + 1)] != broadcasted_shape[broadcasted_rank - (i + 1)])
            {
                ++(*length_keep_dimension);
            }
        }
        else
        {
            ++(*length_remove_dimension);
        }
    }

    return NULL;
}

nw_error_t *reduce_axis(const uint64_t *original_shape,
                        uint64_t original_rank,
                        const uint64_t *broadcasted_shape,
                        uint64_t broadcasted_rank, 
                        uint64_t *axis_keep_dimension,
                        uint64_t *axis_remove_dimension)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape");
    CHECK_NULL_ARGUMENT(axis_keep_dimension, "axis_keep_dimension");
    CHECK_NULL_ARGUMENT(axis_remove_dimension, "axis_remove_dimension");

    if (original_rank > MAX_RANK || broadcasted_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu and broadcasted rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (unsigned long) broadcasted_rank, (int) MAX_RANK), NULL);
    }

    if (!is_broadcastable(original_shape, original_rank, broadcasted_shape, broadcasted_rank))
    {
        return ERROR(ERROR_BROADCAST,
                     string_create("cannot broadcast shapes."),
                     NULL);
    }

    uint64_t j = 0;
    uint64_t k = 0;
    for (uint64_t i = 0; i < broadcasted_rank; ++i)
    {
        if (original_rank >= (i + 1))
        {
            if (original_shape[original_rank - (i + 1)] != broadcasted_shape[broadcasted_rank - (i + 1)])
            {
                axis_keep_dimension[j] = broadcasted_rank - (i + 1);
                ++j;
            }
        }
        else
        {
            axis_remove_dimension[k] = broadcasted_rank - (i + 1);
            ++k;
        }
    }

    return NULL;
}

nw_error_t *slice_shape(const uint64_t *original_shape,
                        uint64_t original_rank,
                        uint64_t *slice_shape,
                        uint64_t slice_rank,
                        const uint64_t *arguments,
                        uint64_t length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(slice_shape, "slice_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    if (original_rank != slice_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with original rank %lu and sliced rank %lu.", 
                     (unsigned long) original_rank, (unsigned long) slice_rank),
                     NULL);
    }

    if (original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (int) MAX_RANK),
                     NULL);
    }

    if (length % 2 != 0 || original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with original rank %lu "
                     "and axis length %lu which should be a multiple of 2.",
                     (unsigned long) original_rank, (unsigned long) length),
                     NULL);
    }

    for (uint64_t i = 0; i < original_rank; ++i)
    {
        if (arguments[2 * i + 1] <= arguments[2 * i] ||
            arguments[2 * i] > original_shape[i] ||
            arguments[2 * i + 1] > original_shape[i])
        {
            return ERROR(ERROR_SHAPE_CONFLICT, 
                         string_create("upperbound of slice %lu must be greater"
                         "than lower bound %lu and bounds must be less than dimension %lu.", 
                         (unsigned long) arguments[2 * i + 1], 
                         (unsigned long) arguments[2 * i],
                         (unsigned long) original_shape[i]),
                         NULL);
        }
        slice_shape[i] = (arguments[2 * i + 1] - arguments[2 * i]); 
    }

    return NULL;
}

nw_error_t *slice_offset(const uint64_t *original_strides,
                         uint64_t original_rank,
                         uint64_t *offset,
                         const uint64_t *arguments,
                         uint64_t length)
{
    CHECK_NULL_ARGUMENT(original_strides, "original_strides");
    CHECK_NULL_ARGUMENT(offset, "offset");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    if (original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (int) MAX_RANK), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks with original rank %lu and axis length %lu which should be a multiple of 2.",
                     (unsigned long) original_rank, (unsigned long) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %lu and length of arguments %lu.",
                     (unsigned long) original_rank, (unsigned long) length), NULL);
    }

    *offset = 0;
    for (uint64_t i = 0; i < length; i += 2)
    {
        *offset += original_strides[i / 2] * arguments[i];
    }

    return NULL;
}

nw_error_t *reverse_slice(const uint64_t *original_shape,
                          uint64_t original_rank,
                          const uint64_t *arguments,
                          uint64_t length,
                          uint64_t *new_arguments,
                          uint64_t new_length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(new_arguments, "new_arguments");

    if (new_length != length)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %lu is not equal to length of new arguments %lu.", 
                     (unsigned long) length, (unsigned long) new_length), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %lu is not a multiple of 2.", 
                     (unsigned long) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %lu and length of arguments %lu.",
                     (unsigned long) original_rank, (unsigned long) length), NULL);
    }

    if (original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (int) MAX_RANK), NULL);
    }

    for (uint64_t i = 0; i < new_length; i += 2)
    {
        new_arguments[i] = arguments[i];
        new_arguments[i + 1] = original_shape[i / 2] - arguments[i + 1];
    }

    return NULL;
}

nw_error_t *padding(const uint64_t *original_shape,
                    uint64_t original_rank,
                    uint64_t *padding_shape,
                    uint64_t padding_rank,
                    const uint64_t *arguments,
                    uint64_t length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(padding_shape, "padding_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    if (original_rank != padding_rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("conflicting ranks with original rank %lu, padding rank %lu.", 
                     (unsigned long) original_rank, (unsigned long) padding_rank), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of arguments %lu is not a multiple of 2.", 
                     (unsigned long) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %lu and length of arguments %lu.",
                     (unsigned long) original_rank, (unsigned long) length), NULL);
    }

    if (original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (int) MAX_RANK), NULL);
    }

    for (uint64_t i = 0; i < original_rank; ++i)
    {
        padding_shape[i] = arguments[2 * i] + arguments[2 * i + 1] + original_shape[i]; 
    }

    return NULL;
}

nw_error_t *reverse_padding(const uint64_t *original_shape, 
                            uint64_t original_rank,
                            const uint64_t *arguments,
                            uint64_t length,
                            uint64_t *new_arguments,
                            uint64_t new_length)
{
    CHECK_NULL_ARGUMENT(original_shape, "original_shape");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(new_arguments, "new_arguments");

    if (new_length != length)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %lu is not equal to length of new arguments %lu.", 
                     (unsigned long) length, (unsigned long) new_length), NULL);
    }

    if (length % 2 != 0)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("length of original arguments %lu is not a multiple of 2.", 
                     (unsigned long) length), NULL);
    }

    if (original_rank != length / 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflict between rank %lu and length of arguments %lu.",
                     (unsigned long) original_rank, (unsigned long) length), NULL);
    }

    if (original_rank > MAX_RANK)
    {
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("original rank %lu must be less than or equal to %d.", 
                     (unsigned long) original_rank, (int) MAX_RANK), NULL);
    }

    for (uint64_t i = 0; i < new_length; i += 2)
    {
        new_arguments[i] = arguments[i];
        new_arguments[i + 1] = original_shape[i / 2] + arguments[i];
    }

    return NULL;
}
