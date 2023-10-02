/**@file random.h
 * @brief Provides probability distribution utilities.
 *
 */

#ifndef RANDOM_H
#define RANDOM_H

#include <datatype.h>
#include <math.h>

void set_seed(uint64_t seed);
float32_t uniformf(float32_t lower_bound, float32_t upper_bound);
float64_t uniform(float64_t lower_bound, float64_t upper_bound);
float32_t normalf(float32_t mean, float32_t standard_deviation);
float64_t normal(float64_t mean, float64_t standard_deviation);
void shuffle_array(int64_t *array, int64_t length);

#endif
