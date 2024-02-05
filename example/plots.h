#ifndef PLOTS_H
#define PLOTS_H

#include <datatype.h>
#include <errors.h>

typedef struct tensor_t tensor_t;

typedef struct pixel_t
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
} pixel_t;

typedef struct bitmap_t
{
    pixel_t *pixels;
    size_t width;
    size_t height;
} bitmap_t;

nw_error_t *bounded_plot(string_t title, string_t save_path, string_t x_str, void* x, int x_n,
                         string_t y_str, void* y, int y_n, float64_t y_min, float64_t y_max, datatype_t datatype);
nw_error_t *plot(string_t title, string_t save_path, string_t x_str, void* x, int x_n, string_t y_str, void* y, int y_n, datatype_t datatype);
nw_error_t *save_png_to_file(bitmap_t *bitmap, string_t path);
nw_error_t *save_tensor_grayscale_to_png_file(tensor_t *tensor, string_t path);
pixel_t *pixel_at(bitmap_t *bitmap, int x, int y);

#endif