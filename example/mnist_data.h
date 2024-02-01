#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <datatype.h>
#include <errors.h>
#include <train.h>

typedef struct mnist_dataset_t
{
    string_t images_path;
    string_t labels_path;
    FILE *images_file;
    FILE *labels_file;
    int64_t height;
    int64_t width;
    int64_t number_of_labels;
    int64_t image_offset;
    int64_t label_offset;
    bool_t normalize;
} mnist_dataset_t;

nw_error_t *mnist_setup(void *arguments);
nw_error_t *mnist_teardown(void *arguments);
nw_error_t *mnist_dataloader(int64_t index, batch_t *batch, void *arguments);
#endif