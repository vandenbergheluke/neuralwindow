/**@file random.h
 * @brief Provides probability distribution utilities.
 *
 */

#ifndef RANDOM_H
#define RANDOM_H

#include <datatype.h>
#include <math.h>

void set_seed(uint64_t seed);
float64_t uniform(float64_t lower_bound, float64_t upper_bound);
float64_t normal(float64_t mean, float64_t variance);

#endif
