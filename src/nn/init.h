#ifndef INIT_H
#define INIT_H

#include <tensor.h>

nw_error_t *init_zeroes(tensor_t *x);
nw_error_t *init_ones(tensor_t *x);

#endif