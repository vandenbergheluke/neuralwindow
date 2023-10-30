#include <graph.h>

static Agraph_t *graph = NULL;
static map_t *map = NULL;
static uint64_t global_node_id = 0;
static FILE *file;

nw_error_t *initialize_graph()
{
    graph = agopen("G", Agstrictdirected, 0);
    if (graph == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, "Graph file could not be created", NULL);
    }

    agattr(graph, AGRAPH, "rankdir", "LR");
    agattr(graph, AGNODE, "shape", "record");

    // Define nodes and labels for the legend
    Agraph_t *legend;
    Agnode_t *red_node, *blue_node;

    legend = agsubg(graph, "Legend", 1);
    red_node = agnode(legend, "red_Node", 1);
    agsafeset(red_node, "label", "Binary: red", "");
    agsafeset(red_node, "shape", "box", "");
    agsafeset(red_node, "color", "red", "");

    blue_node = agnode(legend, "blue_Node", 1);
    agsafeset(blue_node, "label", "Unary: blue", "");
    agsafeset(blue_node, "shape", "box", "");
    agsafeset(blue_node, "color", "blue", "");

    nw_error_t *error = map_create(&map);
    if (error != NULL) 
    {
        agclose(graph);
        return ERROR(ERROR_MEMORY_ALLOCATION, "map could not be created", NULL);
    }

    return NULL;
}

void destroy_graph()
{    
    map_destroy(map);
    // add graph to the dot file
    file = fopen("graph.dot", "w");
    if (file != NULL)
    {
        agwrite(graph, file);
        fclose(file);
    }
    agclose(graph);
    
}

nw_error_t *create_graph_node(graph_node_t **graph_node, uint64_t tensor_id, uint64_t new_graph_id)
{
    CHECK_NULL_ARGUMENT(graph_node, "graph node");

    size_t size = sizeof(graph_node_t);
    *graph_node = (graph_node_t *) malloc(size);
    if (graph_node == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate graph node of size %zu bytes.", size),
                     NULL);
    }
    
    // Initialize
    (*graph_node)->tensor_id = tensor_id;
    (*graph_node)->graph_id = new_graph_id;
    
    return NULL;
}

void get_attribute_format(string_t str, uint64_t rank, uint64_t *attr)
{
    const char* formatted_str = NULL;

    switch(rank)
    {
        case 1:
             formatted_str = string_create("(%d)", *attr);
             break;
        case 2:
            formatted_str = string_create("(%d, %d)", *attr, *(attr + 1));
            break;
        case 3:
            formatted_str =  string_create("(%d, %d, %d)", *attr, *(attr + 1), *(attr + 2));
            break;
        case 4:
            formatted_str = string_create("(%d, %d, %d, %d)", *attr, *(attr + 1), *(attr + 2), *(attr + 3));
            break;
        case 5:
            formatted_str = string_create("(%d, %d, %d, %d, %d)", *attr, *(attr + 1), *(attr + 2), *(attr + 3), *(attr + 4));
            break;
        default:
            return;
    }

    //strcpy(str, formatted_str);
    memcpy((char*)str, formatted_str, strlen(formatted_str));
    string_destroy(formatted_str);
    return;
}

uint64_t graph_tensor_node(tensor_t *tensor)
{
    nw_error_t *error = NULL;
    uint64_t return_value = -1;

    if (graph == NULL || map == NULL)
    {
        error = initialize_graph();
        if (error)
        {
            return -1;
        }
    }

    uint64_t rank = tensor->buffer->view->rank;

    string_t shapeString = string_create("");
    string_t strideString = string_create("");
    get_attribute_format(shapeString, rank, tensor->buffer->view->shape);
    get_attribute_format(strideString, rank, tensor->buffer->view->strides);

    string_t node_label = string_create("<F0> Tensor_ID: %d|Shape: %s|Size: %d|Stride: %s|Offset: %d",
                                        tensor->id,
                                        shapeString,
                                        tensor->buffer->storage->n,
                                        strideString,
                                        tensor->buffer->view->offset);

    // Check if the tensor is already in the map
    void *map_ID;
    string_t map_ID_string = NULL;
    string_t tensor_id = string_create("%ld", tensor->id);
    
    if (!map_contains(map, tensor_id)) {
        map_ID = (void *)(uintptr_t)++global_node_id;

        // Update map
        error = map_set(map, tensor_id, (void *) map_ID);
        if (error)
        {
            string_destroy(tensor_id);
            return_value = -1;
            goto cleanup;
        }
    } 
    else
    {
        // get value from map
        void *prev_global_node_id;
        error = map_get(map, tensor_id, &prev_global_node_id);
        if (error)
        {
            return_value = -1;
            goto cleanup;
        }
        string_destroy(tensor_id);
        map_ID = (void *)(uintptr_t)prev_global_node_id;
    }

    // Update graph
    map_ID_string = string_create("%d", map_ID);
    Agnode_t *node = agnode(graph, (char *)map_ID_string, 1);
    if (node == NULL)
    {
        return_value = -1;
        goto cleanup; 
    }

    agsafeset(node, "label", (char *)node_label, "");

    return_value = (uint64_t) map_ID;

cleanup:
    string_destroy(node_label);
    string_destroy(shapeString);
    string_destroy(strideString);
    if (map_ID_string) string_destroy(map_ID_string);
    return return_value;
}

uint64_t graph_operation_node(string_t op, bool_t binary)
{  
    string_t str1 = string_create("%d", ++global_node_id);
    
    Agnode_t *node = agnode(graph, (char *)str1, 1);
    agsafeset(node, "shape", "oval", "");
    agsafeset(node, "label", (char *)op, "");
    if (binary)
    {
        agsafeset(node, "color", "red", "");
    }
    else
    {
        agsafeset(node, "color", "blue", "");
    }

    string_destroy(str1);
    uint64_t returned_integer = global_node_id;
    return returned_integer;
}

nw_error_t *graph_edge(uint64_t node_1_ID, uint64_t node_2_ID)
{
    Agnode_t *node1, *node2;
    string_t str1 = string_create("%d", node_1_ID);
    string_t str2 = string_create("%d", node_2_ID);

    node1 = agnode(graph, (char *)str1, 0);
    node2 = agnode(graph, (char *)str2, 0);
    if (node1 == NULL || node2 == NULL)
    {
        string_destroy(str1);
        string_destroy(str2);
        return ERROR(ERROR_GRAPH,"Could not add edge on the graph, nodes retreived from graphviz are null", NULL);  
    }

    Agedge_t *edge = agedge(graph, node1, node2, NULL, 1);
    if (edge == NULL) 
    { 
        string_destroy(str1);
        string_destroy(str2);  
        return ERROR(ERROR_GRAPH, "Could not add edge on the graph", NULL); 
    }

    string_destroy(str1);
    string_destroy(str2);
    return NULL;
}

nw_error_t *graph_binary_operation(tensor_t *x, tensor_t *y, tensor_t *z, string_t operation)
{
    uint64_t x_id, y_id, z_id, op_id;
    nw_error_t *error;

    // add nodes
    x_id = graph_tensor_node(x);
    if (x_id == (uint64_t)-1)
    {
        return ERROR(ERROR_GRAPH,"Could not add tensor node to the graph", NULL); 
    }

    y_id = graph_tensor_node(y);
    if (y_id == (uint64_t)-1)
    {
        return ERROR(ERROR_GRAPH,"Could not add tensor node to the graph", NULL); 
    }

    z_id = graph_tensor_node(z);
    if (z_id == (uint64_t)-1)
    {
        return ERROR(ERROR_GRAPH,"Could not add tensor node to the graph", NULL); 
    }

    op_id = graph_operation_node(operation, true);

    // add edges
    error = graph_edge(x_id, op_id);
    if (error != NULL)
    {
        return error;
    }
 
    error = graph_edge(y_id, op_id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_edge(op_id, z_id);
    if (error != NULL)
    {
        return error;
    }

    return NULL;
}

nw_error_t *graph_unary_operation(tensor_t *x, tensor_t *y, string_t operation)
{
    uint64_t x_id, y_id, op_id;
    nw_error_t *error;

    // add node for tensor x
    x_id = graph_tensor_node(x);
    if (x_id == (uint64_t)-1)
    {
        return ERROR(ERROR_GRAPH,"Could not add tensor node to the graph", NULL); 
    }
   
    // add node for tensor y
    y_id = graph_tensor_node(y);
    if (y_id == (uint64_t)-1)
    {
        return ERROR(ERROR_GRAPH,"Could not add tensor node to the graph", NULL); 
    }

    // add node for operation  
    op_id = graph_operation_node(operation, false);

    // add edge between x and operation
    error = graph_edge(x_id, op_id);
    if (error != NULL)
    {
        return error;
    }

    // add edge between operation and y
    error = graph_edge(op_id, y_id);
    if (error != NULL)
    {
        return error;
    }

    return NULL;
}