#include <view.h>

error_t *create_view(view_t **view, uint32_t offset, uint32_t rank, uint32_t *shape, uint32_t *strides)
{
    CHECK_NULL(view, "view");

    error_t *error;
    error = nw_malloc((void **) view, sizeof(view_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create view."), error);

    // Initialize
    (*view)->offset = offset;
    (*view)->rank = rank;
    (*view)->shape = shape;
    (*view)->strides = strides;

    return NULL;
}

error_t *destroy_view(view_t *view)
{
    if (view == NULL)
        return NULL;

    error_t *error;
    error = nw_free(view->shape, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy view->shape."), error);

    error = nw_free(view->strides, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy view->strides."), error);

    error = nw_free(view, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy view."), error);

    return NULL;
}

bool_t is_contiguous(const view_t *view)
{
    if (view == NULL || view->shape == NULL || view->strides == NULL)
        return false;
    
    for (uint32_t i = view->rank - 1; i > 0; i--)
    {
        if ((i == view->rank - 1 && view->strides[i] != 1) || 
            (i < view->rank - 1 && view->strides[i] != view->shape[i + 1] * view->strides[i + 1]))
            return false;
    }

    return true;
}

bool_t equal_shape(const view_t *view_x, const view_t *view_y)
{
    if (view_x == NULL ||
        view_y == NULL ||
        view_x->rank != view_y->rank ||
        view_x->shape == NULL ||
        view_y->shape == NULL)
        return false;

    for (uint32_t i = 0; i < view_x->rank; i++)
    {
        if (view_x->shape[i] != view_y->shape[i])
            return false;
    }

    return true;
}

uint32_t size(const view_t *view)
{
    if (view == NULL || view->shape == NULL)
        return 0;

    uint32_t total = 0;
    for (uint32_t i = 0; i < view->rank; i++)
        total = (i == 0) ? view->shape[i] : total * view->shape[i];
    
    return total;
}

error_t *get_strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank)
{
    CHECK_NULL(strides, "strides");
    CHECK_NULL(shape, "shape");

    for (uint32_t i = rank - 1; i > 0; i--)
    {
        if (i == rank - 1)
            strides[i] = 1;
        else
            strides[i] = shape[i + 1] * strides[i + 1];
    }

    return NULL;
}

error_t *broadcast_shapes(const uint32_t *shape1, uint32_t rank1, const uint32_t *shape2, uint32_t rank2, uint32_t *broadcasted_shape)
{
    CHECK_NULL(shape1, "shape1"); 
    CHECK_NULL(shape2, "shape2"); 
    CHECK_NULL(broadcasted_shape, "broadcasted_shape"); 

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
            return ERROR(ERROR_BROADCAST, create_string("failed to broadcast shapes"), NULL);
        }
    }
    return NULL;
}
