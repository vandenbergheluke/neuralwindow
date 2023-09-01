/**@file init.h
 * @brief
 *
 */

#ifndef INIT_H
#define INIT_H

#include <errors.h>

// Forward declarations
typedef struct tensor_t tensor_t;

nw_error_t *init_zeroes(tensor_t *x);
nw_error_t *init_ones(tensor_t *x);

#endif
