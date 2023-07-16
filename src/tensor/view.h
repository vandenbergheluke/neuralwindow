#ifndef VIEW_H
#define VIEW_H

#include <nw_runtime.h>
#include <datatype.h>
#include <errors.h>

typedef struct view_t
{
    uint32_t *shape;
    uint32_t rank; 
    uint32_t *strides;
    uint32_t offset;
} view_t;

error_t *create_view(view_t **view, uint32_t offset, uint32_t rank, uint32_t *shape, uint32_t *strides);
error_t *destroy_view(view_t *view);

bool_t is_contiguous(const view_t *view);
bool_t equal_shape(const view_t *view_x, const view_t *view_y);
uint32_t size(const view_t *view);
error_t *get_strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank);


#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif