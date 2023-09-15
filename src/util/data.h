#ifndef DATA_H
#define DATA_H

#include <datatype.h>

typedef struct tensor_t tensor_t;

typedef struct dataset_t
{
    uint64_t batch_size;
    uint64_t number_of_samples;
    bool_t shuffle; 
    datatype_t datatype;
    runtime_t runtime;
    void *arguments;
} dataset_t;
    
typedef struct batch_t
{
    tensor_t *x;
    tensor_t *y;
} batch_t;


#endif