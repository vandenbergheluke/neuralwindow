#include <view.h>

error_t *view_create(view_t **view, uint32_t offset, uint32_t rank, const uint32_t *shape, const uint32_t *strides)
{
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(shape, "shape");

    size_t size = sizeof(view_t);
    *view = (view_t *) malloc(size);
    if (view == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view of size %zu.", size),
                     NULL);
    }

    size = rank * sizeof(uint32_t);
    (*view)->shape = (uint32_t *) malloc(size);
    if ((*view)->shape == NULL)
    {
        free(*view);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view->shape of size %zu.", size),
                     NULL);
    }
    memcpy((*view)->shape, shape, size);

    (*view)->strides = (uint32_t *) malloc(size);
    if ((*view)->strides == NULL)
    {
        free(*view);
        free((*view)->shape);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate view->strides of size %zu.", size),
                     NULL);
    }
    if (strides != NULL)
    {
        memcpy((*view)->strides, strides, size);
    }
    else
    {
        error_t *error = get_strides_from_shape((*view)->strides, shape, rank);
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

    (*view)->offset = offset;
    (*view)->rank = rank;

    return NULL;
}

void view_destroy(view_t *view)
{
    if (view != NULL)
    {
        free(view->shape);
        free(view->strides);
        free(view);
    }
}

bool_t view_is_contiguous(const view_t *view)
{
    if (view == NULL || view->shape == NULL || view->strides == NULL)
    {
        return false;
    }
    
    for (uint32_t i = view->rank - 1; i > 0; i--)
    {
        if ((i == view->rank - 1 && view->strides[i] != 1) || 
            (i < view->rank - 1 && view->strides[i] != view->shape[i + 1] * view->strides[i + 1]))
        {
            return false;
        }
    }

    return true;
}

bool_t view_shape_equal(const view_t *view_x, const view_t *view_y)
{
    if (view_x == NULL ||
        view_y == NULL ||
        view_x->rank != view_y->rank ||
        view_x->shape == NULL ||
        view_y->shape == NULL)
    {
        return false;
    }

    for (uint32_t i = 0; i < view_x->rank; i++)
    {
        if (view_x->shape[i] != view_y->shape[i])
        {
            return false;
        }
    }

    return true;
}

uint32_t view_size(const view_t *view)
{
    if (view == NULL || view->shape == NULL)
    {
        return 0;
    }

    uint32_t total = 0;
    for (uint32_t i = 0; i < view->rank; i++)
    {
        total = (i == 0) ? view->shape[i] : total * view->shape[i];
    }
    
    return total;
}

error_t *get_strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank)
{
    CHECK_NULL_ARGUMENT(strides, "strides");
    CHECK_NULL_ARGUMENT(shape, "shape");

    for (uint32_t i = rank - 1; i > 0; i--)
    {
        if (i == rank - 1)
        {
            strides[i] = 1;
        }
        else
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    return NULL;
}

error_t *broadcast_shapes(const uint32_t *shape1, uint32_t rank1, const uint32_t *shape2, uint32_t rank2, uint32_t *broadcasted_shape)
{
    CHECK_NULL_ARGUMENT(shape1, "shape1"); 
    CHECK_NULL_ARGUMENT(shape2, "shape2"); 
    CHECK_NULL_ARGUMENT(broadcasted_shape, "broadcasted_shape"); 

    uint32_t rank = MAX(rank1, rank2);
    for (uint32_t i = rank - 1; i > 0; i--)
    {
        if (i >= rank1 || (i < rank2 && shape1[i] == 1))
        {
            broadcasted_shape[i] = shape2[i];
        } 
        else if (i >= rank2 || shape1[i] == shape2[i] || shape2[i] == 1)
        {
            broadcasted_shape[i] = shape1[i];
        }
        else
        {
            return ERROR(ERROR_BROADCAST, 
                         string_create("failed to broadcast shapes."),
                         NULL);
        }
    }
    return NULL;
}
