#ifndef VIEW_H
#define VIEW_H

#include <datatype.h>
#include <errors.h>

#define MAX_RANK 5

typedef struct view_t
{
    uint32_t *shape;
    uint32_t rank; 
    uint32_t *strides;
    uint32_t offset;
} view_t;

error_t *view_create(view_t **view, uint32_t offset, uint32_t rank, const uint32_t *shape, const uint32_t *strides);
void view_destroy(view_t *view);
bool_t view_is_contiguous(const view_t *view);
bool_t view_shape_equal(const view_t *view_x, const view_t *view_y);
uint32_t view_size(const view_t *view);
error_t *get_strides_from_shape(uint32_t *strides, const uint32_t *shape, uint32_t rank);

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif