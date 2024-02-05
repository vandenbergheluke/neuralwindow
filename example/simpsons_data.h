#ifndef SIMPSONS_DATA_H
#define SIMPSONS_DATA_H

#include <datatype.h>
#include <errors.h>
#include <train.h>

typedef struct simpsons_dataset_t
{
    string_t data_path;
    FILE *data_file;
    int64_t vocabulary_size;
    char integer_to_character[CHAR_MAX];
    int64_t character_to_integer[CHAR_MAX];
    int64_t number_of_characters;
    int64_t block_size;
} simpsons_dataset_t;

nw_error_t *simpsons_setup(void *arguments);
nw_error_t *simpsons_teardown(void *arguments);
nw_error_t *simpsons_dataloader(int64_t index, batch_t *batch, void *arguments);


#endif