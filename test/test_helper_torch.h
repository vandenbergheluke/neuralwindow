#ifndef TEST_HELPER_TORCH
#define TEST_HELPER_TORCH

#include <torch/torch.h>

typedef struct tensor_t tensor_t;
typedef enum runtime_t runtime_t;
typedef enum datatype_t datatype_t;

#define SEED 1234

tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype);

#endif