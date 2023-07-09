#include <view.h>

error_t *create_view(view_t **view, runtime_t runtime)
{
    CHECK_NULL_POINTER(view, "view");

    error_t *error = nw_malloc((void **) view, sizeof(view_t), runtime);
    if (error != NULL)
    {
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, create_string("failed to create view."), error);
    }
    return NULL;
}

error_t *destroy_view(view_t *view, runtime_t runtime)
{
    CHECK_NULL_POINTER(view, "view");

    error_t *error = nw_free(view->shape, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy view."), error);
    }

    error = nw_free(view->strides, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy view."), error);
    }

    error = nw_free(view, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy view."), error);
    }
    return NULL;
}

error_t *get_strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank)
{
    CHECK_NULL_POINTER(strides, "strides");
    CHECK_NULL_POINTER(shape, "shape");

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
    CHECK_NULL_POINTER(shape1, "shape1"); 
    CHECK_NULL_POINTER(shape2, "shape2"); 
    CHECK_NULL_POINTER(broadcasted_shape, "broadcasted_shape"); 

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
            return create_error(ERROR_BROADCAST, __FILE__, __LINE__, __FUNCTION__, NULL, NULL);
        }
    }
    return NULL;
}
