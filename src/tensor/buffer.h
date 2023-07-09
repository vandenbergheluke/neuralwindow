#ifndef BUFFER_H
#define BUFFER_H

#include <nw_runtime.h>
#include <view.h>

typedef struct buffer_t
{
    view_t *view;
    runtime_t runtime;
    datatype_t datatype;
    void *data;
} buffer_t;

error_t *create_buffer(buffer_t **buffer, runtime_t runtime);
error_t *destroy_buffer(buffer_t *buffer, runtime_t runtime);

#endif