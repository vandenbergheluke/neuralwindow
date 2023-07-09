#include <layer.h>
#include <nw_runtime.h>

error_t *initialize(module_t *module, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_POINTER(module, "module");
    CHECK_NULL_POINTER(module->units, "model->units");
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
                string_t message = create_string("unknown instance type %d", module->units[i]->layer_type);
                error = create_error(ERROR_UNKNOWN_INSTANCE_TYPE, __FILE__, __LINE__, __FUNCTION__, message, NULL);
                break;
        }
        if (error != NULL)
        {
            string_t message = create_string("failed to initialize module");
            return create_error(ERROR_INITIALIZATION, __FILE__, __LINE__, __FUNCTION__, message, error);
        }
    }
    return NULL;
}

error_t *initialize_linear(linear_t *linear, datatype_t datatype, runtime_t runtime)
{
    return NULL;
}

