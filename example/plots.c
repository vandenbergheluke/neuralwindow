#include <plots.h>
#include <tensor.h>
#include <buffer.h>
#include <view.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <png.h>
#include <mgl2/base_cf.h>
#include <mgl2/canvas_cf.h>
#include <mgl2/mgl_cf.h>

nw_error_t *bounded_plot(string_t title, string_t save_path, string_t x_str, void* x, int x_n,
                         string_t y_str, void* y, int y_n, float64_t y_min, float64_t y_max, datatype_t datatype)
{
    HMGL graph = mgl_create_graph(800,400);

    HMDT x_mgl = mgl_create_data();
    HMDT y_mgl = mgl_create_data();

    mgl_data_set_float(x_mgl, x, x_n, 1, 1);
    switch (datatype)
    {
    case FLOAT32:
        mgl_data_set_float(y_mgl, (float32_t *) y, y_n, 1, 1);
        break;
    case FLOAT64:
        mgl_data_set_double(y_mgl, (float64_t *) y, y_n, 1, 1);
        break;
    default:
        mgl_delete_graph(graph);

        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
    }

    mgl_fill_background(graph, 1, 1, 1, 1);

    //mgl_subplot(graph, 3, 3, 4, "");
    mgl_inplot(graph, 0, 1, 0, 1);
    mgl_title(graph, title, "", 5);
    mgl_set_range_dat(graph, 'x', x_mgl, 0);
    mgl_set_range_val(graph, 'y', y_min, y_max);
    mgl_axis(graph, "xy", "", "");
    // |    long dashed line
    // h    grey
    mgl_axis_grid(graph, "xy", "|h", "");
    mgl_label(graph, 'x', x_str, 0, "");
    mgl_label(graph, 'y', y_str, 0, "");
    mgl_box(graph);
    // u    blue purple
    mgl_plot_xy(graph, x_mgl, y_mgl, "2u", "");

    mgl_write_png(graph, save_path, "w");

    mgl_delete_graph(graph);
    mgl_delete_data(x_mgl);
    mgl_delete_data(y_mgl);

    return NULL;
}

nw_error_t *plot(string_t title, string_t save_path, string_t x_str, void* x, int x_n, string_t y_str, void* y, int y_n, datatype_t datatype)
{
    HMGL graph = mgl_create_graph(800,400);

    HMDT x_mgl = mgl_create_data();
    HMDT y_mgl = mgl_create_data();

    mgl_data_set_float(x_mgl, x, x_n, 1, 1);
    switch (datatype)
    {
    case FLOAT32:
        mgl_data_set_float(y_mgl, (float32_t *) y, y_n, 1, 1);
        break;
    case FLOAT64:
        mgl_data_set_double(y_mgl, (float64_t *) y, y_n, 1, 1);
        break;
    default:
        mgl_delete_graph(graph);

        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
    }

    mgl_fill_background(graph, 1, 1, 1, 1);

    //mgl_subplot(graph, 3, 3, 4, "");
    mgl_inplot(graph, 0, 1, 0, 1);
    mgl_title(graph, title, "", 5);
    mgl_set_range_dat(graph, 'x', x_mgl, 0);
    mgl_set_range_dat(graph, 'y', y_mgl, 0);
    mgl_axis(graph, "xy", "", "");
    // |    long dashed line
    // h    grey
    mgl_axis_grid(graph, "xy", "|h", "");
    mgl_label(graph, 'x', x_str, 0, "");
    mgl_label(graph, 'y', y_str, 0, "");
    mgl_box(graph);
    // u    blue purple
    mgl_plot_xy(graph, x_mgl, y_mgl, "2u", "");

    mgl_write_png(graph, save_path, "w");

    mgl_delete_graph(graph);
    mgl_delete_data(x_mgl);
    mgl_delete_data(y_mgl);

    return NULL;
}

nw_error_t *save_tensor_grayscale_to_png_file(tensor_t *tensor, string_t path)
{
    bitmap_t png_img;

    png_img.width = tensor->buffer->view->shape[3];
    png_img.height =  tensor->buffer->view->shape[2];

    png_img.pixels = (pixel_t *) calloc(sizeof(pixel_t), png_img.width * png_img.height);
    
    for (size_t x = 0; x < png_img.width; x++)
    {
        for (size_t y = 0; y < png_img.height; y++)
        {
            pixel_t* pixel = pixel_at(&png_img, x, y);

            uint8_t v = ((((float32_t *) tensor->buffer->storage->data)[png_img.width * y + x] * 0.5) + 0.5) * 255;
            pixel->red = v;
            pixel->green = v;
            pixel->blue = v;
            pixel->alpha = 255;

        }
    }

    save_png_to_file(&png_img, path);

    free(png_img.pixels);

    return NULL;
}


pixel_t *pixel_at(bitmap_t *bitmap, int x, int y)
{
    return bitmap->pixels + bitmap->width * y + x;
}

nw_error_t *save_png_to_file(bitmap_t *bitmap, string_t path)
{
    FILE *fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_byte **row_pointers = NULL;
    
    int pixel_size = 4;
    int depth = 8;

    fp = fopen(path, "wb");
    if (!fp)
    {
        return ERROR(ERROR_FILE, string_create("failed to open file."), NULL);
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) 
    {
        fclose(fp);
        return ERROR(ERROR_CREATE, string_create("failed to create write struct."), NULL);
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return ERROR(ERROR_CREATE, string_create("failed to create info struct."), NULL);
    }

    if (setjmp(png_jmpbuf(png_ptr))) 
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return ERROR(ERROR_SET, string_create("failed to set jump to start."), NULL);
    }

    png_set_IHDR(png_ptr, info_ptr, bitmap->width, bitmap->height, depth, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    row_pointers = (png_byte **) png_malloc(png_ptr, bitmap->height * sizeof(png_byte*));

    for (size_t y = 0; y < bitmap->height; ++y)
    {
        png_byte *row = (png_byte *)( png_malloc(png_ptr, sizeof(uint8_t) * bitmap->width * pixel_size));
        row_pointers[y] = row;
        for (size_t x = 0; x < bitmap->width; ++x)
        {
            pixel_t *pixel = pixel_at(bitmap, x, y);
            *row++ = pixel->red;
            *row++ = pixel->green;
            *row++ = pixel->blue;
            *row++ = pixel->alpha;
        }

    }

    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    for (size_t y = 0; y < bitmap->height; y++)
    {
        png_free(png_ptr, row_pointers[y]);
    }
   
    png_free(png_ptr, row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return NULL;
}


// int main()
// {
//     bitmap_t png_img;

//     png_img.width = 400;
//     png_img.height = 400;

//     png_img.pixels = static_cast<pixel_t*>(calloc(sizeof(pixel_t), png_img.width * png_img.height));
    
//     for (int x = 0; x < png_img.width; x++) {
//         for (int y = 0; y < png_img.height; y++) {

//             pixel_t* pixel = pixel_at(&png_img, x, y);

//             pixel->red = (y + 80) % 170;
//             pixel->green = (y + 11) % 120;
//             pixel->blue = (x + 30) % 255;
//             pixel->alpha = (x * 4) % 255;

//         }
//     }

//     save_png_to_file(&png_img, "..\\image.png");

//     free(png_img.pixels);

//     return 0;
// }