#include <datatype.h>

char *datatype_string(datatype_t datatype)
{
    switch (datatype)
    {
    case FLOAT32:
        return "float32";
    default:
        return "datatype_unknown";
    }
}