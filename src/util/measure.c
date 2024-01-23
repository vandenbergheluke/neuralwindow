/**@file measure.c
 * @brief Implements utilities for taking measurements.
 *
 */

#include <measure.h>
#include <time.h>

int64_t get_time_nanoseconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (int64_t) (ts.tv_sec * 1e9) + (int64_t) ts.tv_nsec;
}
