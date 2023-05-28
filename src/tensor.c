#include <tensor.h>
#include <errors.h>
#include <stdfloat.h>

tensor_t *construct_tensor(buffer_t data, datatype_t datatype, shape_t shape)
{
    tensor_t *x = (tensor_t *) malloc(sizeof(tensor_t));
    CHECK_MEMORY_ALLOCATED(x);
    x->data = data;
    x->datatype = datatype;
    x->shape = shape;
    return x;
}

void destroy_tensor(tensor_t *x) {
    if (x == NULL) {
        return;
    }
    free(x);
}