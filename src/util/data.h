#ifndef DATA_H
#define DATA_H

#include <datatype.h>
#include <buffer.h>

typedef struct tensor_t tensor_t;

typedef enum dataset_type_t
{
    TRAIN,
    VALID,
    TEST
} dataset_type_t;

typedef struct dataset_t
{
    uint64_t batch_size;
    uint64_t number_of_samples;
    bool_t shuffle; 
    datatype_t datatype;
    runtime_t runtime;
    void *arguments;
    float32_t train_split;
    float32_t valid_split;
} dataset_t;
    
typedef struct batch_t
{
    tensor_t *x;
    tensor_t *y;
} batch_t;


#endif