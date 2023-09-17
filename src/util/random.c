/**@file random.c
 * @brief Implements probability distribution utilities.
 *
 */

#include <random.h>

void set_seed(uint64_t seed)
{
    srand(seed);
}

inline float64_t uniform(float64_t lower_bound, float64_t upper_bound)
{
    return (float64_t) rand() / (float64_t) RAND_MAX * (upper_bound - lower_bound) + lower_bound;
}

float64_t normal(float64_t mean, float64_t standard_deviation)
{
    float64_t u, v, r2, f;
    static float64_t sample;
    static bool_t sample_available = false;

    if (sample_available)
    {
        sample_available = false;
        return mean + standard_deviation * sample;
    }

    do
    {
        u = 2.0 * uniform(0.0, 1.0) - 1.0;
        v = 2.0 * uniform(0.0, 1.0) - 1.0;
        r2 = u * u + v * v;
    }
    while (r2 >= 1.0 || r2 == 0.0);

    f = sqrt(-2.0 * log(r2) / r2);
    sample = u * f;
    sample_available = true;

    return mean + standard_deviation * v * f;
}

void shuffle(uint64_t *array, uint64_t length)
{   
    for (uint64_t i = 0; i < length - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (length - i) + 1);
        uint64_t temp = array[j];
        array[j] = array[i];
        array[i] = temp;
    }
}