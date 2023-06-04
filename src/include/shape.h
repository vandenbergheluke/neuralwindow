#ifndef SHAPE_H
#define SHAPE_H

#include <stdint.h>
#include <errors.h>
#include <device.h>
#include <stdbool.h>

#define MAX_RANK 5

typedef struct shape_t
{
    uint32_t *dimensions;
    uint32_t rank;
    uint32_t *stride;
    uint32_t offset;
} shape_t;

error_t *create_shape(shape_t **s, device_t device);
error_t *destroy_shape(shape_t *s, device_t device);
bool shape_equal(shape_t x_shape, shape_t y_shape);
char *shape_string(const shape_t *shape);

#endif