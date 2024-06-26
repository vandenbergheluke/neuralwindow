/**@file random.c
 * @brief Implements probability distribution utilities.
 *
 */

#include <random.h>

void set_seed(uint64_t seed)
{
    srand(seed);
}

float32_t multinomialf(float32_t *probabilities, int64_t length)
{
    float32_t cumsum[length];
    for (int64_t i = 0; i < length; ++i)
    {
        if (!i)
        {
            cumsum[i] = probabilities[i];
        }
        else
        {
            cumsum[i] = probabilities[i] + cumsum[i - 1];
        }
    }

    float32_t x = uniformf(0.0, 1.0);
    for (int64_t i = 0; i < length; ++i)
    {
        if (x <= cumsum[i])
        {
            return (float32_t) i;
        }
    }

    return (float32_t) (length - 1);
}

float64_t multinomial(float64_t *probabilities, int64_t length)
{
    float64_t cumsum[length];
    for (int64_t i = 0; i < length; ++i)
    {
        if (!i)
        {
            cumsum[i] = probabilities[i];
        }
        else
        {
            cumsum[i] = probabilities[i] + cumsum[i - 1];
        }
    }

    float64_t x = uniform(0.0, 1.0);
    for (int64_t i = 0; i < length; ++i)
    {
        if (x <= cumsum[i])
        {
            return (float64_t) i;
        }
    }

    return (float64_t) (length - 1);
}

inline float32_t uniformf(float32_t lower_bound, float32_t upper_bound)
{
    return (float32_t) uniform((float64_t) lower_bound, (float64_t) upper_bound);
}

inline float64_t uniform(float64_t lower_bound, float64_t upper_bound)
{
    return (float64_t) rand() / (float64_t) RAND_MAX * (upper_bound - lower_bound) + lower_bound;
}


float32_t normalf(float32_t mean, float32_t standard_deviation)
{
    return (float32_t) normal((float64_t) mean, (float64_t) standard_deviation);
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

void shuffle_array(int64_t *array, int64_t length)
{   
    for (int64_t i = 0; i < length - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (length - i) + 1);
        int64_t temp = array[j];
        array[j] = array[i];
        array[i] = temp;
    }
}