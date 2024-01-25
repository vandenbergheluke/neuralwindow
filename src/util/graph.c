#include <graph.h>
#include <graphviz/gvc.h>

static Agraph_t *graph = NULL;
static map_t *map = NULL;
static uint64_t global_node_id = 0;
static uint64_t graph_v = 0;
static FILE *file = NULL;

static void add_legend_entry(Agraph_t *legend, operation_type_t operation_type, string_t color)
{
    string_t label = operation_type_string(operation_type);
    Agnode_t *node = agnode(legend, (char *) label, 1);
    agsafeset(node, "label", (char *) label, "");
    agsafeset(node, "shape", "box", "");
    agsafeset(node, "color", (char *) color, "");
}

static void initialize_legend(void)
{
    Agraph_t *legend;
    legend = agsubg(graph, "Legend", 1);
    add_legend_entry(legend, UNARY_OPERATION, "orange");
    add_legend_entry(legend, BINARY_OPERATION, "green");
    add_legend_entry(legend, REDUCTION_OPERATION, "red");
    add_legend_entry(legend, STRUCTURE_OPERATION, "blue");
}

nw_error_t *start_graph(void)
{
    char_t* graph_var = getenv("GRAPH");
    if (graph_var && strcmp(graph_var, "1") == 0)
    {
        graph = agopen("G", Agstrictdirected, 0);
        if (graph == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, "Graph file could not be created", NULL);
        }

        agattr(graph, AGRAPH, "rankdir", "LR");
        agattr(graph, AGNODE, "shape", "record");

        initialize_legend();

        // Storage of graph nodes
        nw_error_t *error = map_create(&map);
        if (error != NULL) 
        {
            agclose(graph);
            return ERROR(ERROR_MEMORY_ALLOCATION, "map could not be created", NULL);
        }
    }
    return NULL;
}

void end_graph(void)
{   
    if (graph)
    {
        map_destroy(map);
        string_t filename = string_create("graph_%lu.dot", graph_v++);
        file = fopen(filename, "w");
        if (file != NULL)
        {
            agwrite(graph, file);
            fclose(file);
        }
        agclose(graph);
        string_destroy(filename);

        graph = NULL;
        map = NULL;
        file = NULL;
    }
}

static string_t int64_array_to_string(const int64_t *array, int64_t length)
{
    if (length < 0 || length > MAX_RANK)
    {
        return NULL;
    }

    switch(length)
    {
    case 0:
        return string_create("()");
    case 1:
        return string_create("(%ld)", array[0]);
    case 2:
        return string_create("(%ld, %ld)", array[0], array[1]);
    case 3:
        return string_create("(%ld, %ld, %ld)", array[0], array[1], array[2]);
    case 4:
        return string_create("(%ld, %ld, %ld, %ld)", array[0], array[1], array[2], array[3]);
    case 5:
        return string_create("(%ld, %ld, %ld, %ld, %ld)", array[0], array[1], array[2], array[3], array[4]);
    }

    return NULL;
}

nw_error_t *graph_tensor_node(tensor_t *tensor, Agnode_t **node)
{
    nw_error_t *error = NULL;
    uint64_t tensor_id = tensor->id;
    int64_t rank = tensor->buffer->view->rank;
    int64_t n = tensor->buffer->storage->n;
    int64_t offset = tensor->buffer->view->offset;
    string_t tensor_id_string = string_create("%lu", tensor->id);

    if (!map_contains(map, tensor_id_string)) {
        string_t node_id_string = string_create("%lu", global_node_id++);
        string_t shape_string = int64_array_to_string(tensor->buffer->view->shape, rank);
        string_t stride_string = int64_array_to_string(tensor->buffer->view->strides, rank);
        string_t node_label = string_create("<F0> Tensor_ID: %lu|Shape: %s|Size: %ld|Stride: %s|Offset: %ld|Requires Gradient: %s", 
                                             tensor_id, shape_string, n, stride_string, offset, (tensor->requires_gradient) ? "true" : "false");
        string_destroy(shape_string);
        string_destroy(stride_string);
        *node = agnode(graph, (char *) node_id_string, 1);
        agsafeset(*node, "label", (char *)node_label, "");

        error = map_set(map, tensor_id_string, (void *) *node);
        if (error)
        {
            string_destroy(tensor_id_string);
            string_destroy(node_id_string);
            string_destroy(node_label);
            return ERROR(ERROR_SET, string_create("failed to set map entry."), error);
            
        }
        string_destroy(node_id_string);
        string_destroy(node_label);
    } 
    else
    {
        error = map_get(map, tensor_id_string, (void **) node);
        if (error)
        {
            string_destroy(tensor_id_string);
            return ERROR(ERROR_SET, string_create("failed to set map entry."), error);
            
        }
        string_destroy(tensor_id_string);
    }

    return error;
}

void graph_function_node(function_t *function, Agnode_t **node)
{  
    string_t name = string_create("%lu", global_node_id++);
    *node = agnode(graph, (char *) name, 1);
    string_destroy(name);
    string_t label, color, arguments;
    switch (function->operation_type)
    {
    case UNARY_OPERATION:
        label = string_create("<F0> Type: %s|Operation: %s", 
                              operation_type_string(function->operation_type),
                              unary_operation_type_string(function->operation->unary_operation->operation_type));
        color = "orange";
        break;
    case BINARY_OPERATION:
        label = string_create("<F0> Type: %s|Operation: %s", 
                              operation_type_string(function->operation_type),
                              binary_operation_type_string(function->operation->binary_operation->operation_type));
        color = "green";
        break;
    case REDUCTION_OPERATION:
        arguments = int64_array_to_string(function->operation->reduction_operation->axis, 
                                          function->operation->reduction_operation->length);
        label = string_create("<F0> Type: %s|Operation: %s|Axis: %s|Keep Dimensions: %s", 
                              operation_type_string(function->operation_type),
                              reduction_operation_type_string(function->operation->reduction_operation->operation_type),
                              arguments, (function->operation->reduction_operation->keep_dimension) ? "true" : "false");
        string_destroy(arguments);
        color = "red";
        break;
    case STRUCTURE_OPERATION:
        arguments = int64_array_to_string(function->operation->structure_operation->arguments, 
                                          function->operation->structure_operation->length);
        label = string_create("<F0> Type: %s|Operation: %s|Arguments: %s", 
                              operation_type_string(function->operation_type), 
                              structure_operation_type_string(function->operation->structure_operation->operation_type),
                              arguments);
        string_destroy(arguments);
        color = "blue";
        break;
    default:
        return;
    }
    agsafeset(*node, "label", (char *) label, "");
    agsafeset(*node, "color", (char *) color, "");
    string_destroy(label);
}

nw_error_t *graph_function(function_t *function, tensor_t *z)
{
    if (!graph)
    {
        return NULL;
    }

    nw_error_t *error = NULL;
    Agnode_t *node_x, *node_y, *node_z, *function_node;

    switch (function->operation_type)
    {
    case UNARY_OPERATION:
        graph_function_node(function, &function_node);
        error = graph_tensor_node(function->operation->unary_operation->x, &node_x);
        if (error)
        {
            return ERROR(ERROR_GRAPH, string_create("failed to graph tensor node."), NULL);
        }
        agedge(graph, node_x, function_node, NULL, 1);
        break;
    case BINARY_OPERATION:
        graph_function_node(function, &function_node);
        error = graph_tensor_node(function->operation->binary_operation->x, &node_x);
        if (error)
        {
            return ERROR(ERROR_GRAPH, string_create("failed to graph tensor node."), NULL);
        }
        agedge(graph, node_x, function_node, NULL, 1);
        error = graph_tensor_node(function->operation->binary_operation->y, &node_y);
        if (error)
        {
            return ERROR(ERROR_GRAPH, string_create("failed to graph tensor node."), NULL);
        }
        agedge(graph, node_y, function_node, NULL, 1);
        break;
    case REDUCTION_OPERATION:
        graph_function_node(function, &function_node);
        error = graph_tensor_node(function->operation->reduction_operation->x, &node_x);
        if (error)
        {
            return ERROR(ERROR_GRAPH, string_create("failed to graph tensor node."), NULL);
        }
        agedge(graph, node_x, function_node, NULL, 1);
        break;
    case STRUCTURE_OPERATION:
        graph_function_node(function, &function_node);
        error = graph_tensor_node(function->operation->structure_operation->x, &node_x);
        if (error)
        {
            return ERROR(ERROR_GRAPH, string_create("failed to graph tensor node."), NULL);
        }
        agedge(graph, node_x, function_node, NULL, 1);
        break;
    default:
        return error;
    }

    error = graph_tensor_node(z, &node_z);
    if (error)
    {
        return ERROR(ERROR_GRAPH, string_create("failed to graph tensor node."), NULL);
    }
    agedge(graph, function_node, node_z, NULL, 1);

    return error;
}