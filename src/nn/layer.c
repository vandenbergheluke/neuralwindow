#include <layer.h>
#include <nw_runtime.h>


error_t *initialize(module_t *module, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL(module, "module");
    CHECK_NULL(module->units, "model->units");
    for (int i = 0; i < module->depth; i++)
    {
        error_t *error;
        switch (module->units[i]->layer_type)
        {
            case MODULE:
                error = initialize(module->units[i]->layer->module, datatype, runtime);
                break;
            case LINEAR:
                error = initialize_linear(module->units[i]->layer->linear, datatype, runtime);
            default:
                error = ERROR(ERROR_UNKNOWN_INSTANCE_TYPE, create_string("unknown instance type %d", module->units[i]->layer_type), NULL);
                break;
        }
        if (error != NULL)
            return ERROR(ERROR_INITIALIZATION, create_string("failed to initialize module"), error);
    }
    return NULL;
}

error_t *initialize_linear(linear_t *linear, datatype_t datatype, runtime_t runtime)
{
    return NULL;
}

