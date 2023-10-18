#include <graph.h>

static Agraph_t *graph = NULL;
static map_t *map = NULL;
static uint64_t node_id = 0;
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
    agclose(graph);
    map_destroy(map);
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

void get_attribute_format(string_t* str, uint64_t rank, uint64_t *attr)
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

    strcpy(str, formatted_str);
    free(formatted_str);
    return;
}

nw_error_t *graph_tensor_node(tensor_t *tensor)
{   
    nw_error_t *error;
    if (graph == NULL || map == NULL)
    {
        
        error = initialize_graph();
        if (error != NULL)
        {
            return error;
        }
        
        error = NULL;
    }

    uint64_t rank = tensor->buffer->view->rank;

    string_t *str1 = string_create(""); 
    string_t *str2 = string_create("");
    get_attribute_format(str1, rank, tensor->buffer->view->shape);
    get_attribute_format(str2, rank, tensor->buffer->view->strides);

    string_t node_label = string_create("<F0> Tensor_ID: %d|Shape: %s|Size: %d|Stride: %s|Offset: %d",  tensor->id,
                                                                                  str1,
                                                                                  tensor->buffer->storage->n,
                                                                                  str2,
                                                                                  tensor->buffer->view->offset);
    
    // check if there is already a node associated with the tensor ID on the graph
    uint64_t new_node_it;
    string_t tensor_id = string_create("%d", tensor->id);
    if (!map_contains(map, tensor_id))
    {
        new_node_it = ++node_id;
        // update map
        error = map_set(map, tensor_id, (void *)new_node_it);
        if (error)
        {
            free(tensor_id);
            free(node_label);
            free(str1);
            free(str2);
            return ERROR(ERROR_GRAPH,
                        string_create("failed to set map entry with corresponding key %d.", tensor->id),
                        error);
        }
    }
    else
    {
        void *prev_node_id;
        error = map_get(map, tensor_id, &prev_node_id);
        if (error)
        {
            free(tensor_id);
            free(node_label);
            free(str1);
            free(str2);
            return ERROR(ERROR_GRAPH, string_create("Could not get tensor node id from the map for tensor: %d", tensor->id), NULL); 
        }
        new_node_it = (uint64_t)prev_node_id;
    }

    // update graph   
    string_t new_node_id = string_create("%d", new_node_it);                                                                     
    Agnode_t *node = agnode(graph, (char *)new_node_id, 1);
    if (node == NULL) 
    {
        free(tensor_id);
        free(node_label);
        free(str1);
        free(str2);
        free(new_node_id);
        return ERROR(ERROR_GRAPH, string_create("Could not add tensor node on the graph for tensor: %d", tensor->id), NULL); 
    }
    agsafeset(node, "label", (char *)node_label, "");

    // free strings
    free(tensor_id);
    free(node_label);
    free(str1);
    free(str2);
    free(new_node_id);
    return NULL;
}

uint64_t graph_operation_node(string_t op, bool_t binary)
{   
    string_t str = string_create("%d", ++node_id);

    Agnode_t *node = agnode(graph, (char *)str, 1);
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

    free(str);
    return node_id;
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
        free(str1);
        free(str2);
        return ERROR(ERROR_GRAPH, "Could not add edge on the graph, nodes retreived from graphviz are null", NULL);  
    }

    Agedge_t *edge = agedge(graph, node1, node2, NULL, 1);
    if (edge == NULL) 
    { 
        free(str1);
        free(str2);  
        return ERROR(ERROR_GRAPH, "Could not add edge on the graph", NULL); 
    }

    free(str1);
    free(str2);
    return NULL;
}


nw_error_t *graph_binary_operation(tensor_t *x, tensor_t *y, tensor_t *z, string_t operation)
{
    uint64_t *x_id = NULL, *y_id = NULL, *z_id = NULL, op_id;
    nw_error_t *error;

    // add nodes
    error = graph_tensor_node(x);
    if (error != NULL)
    {
        return error;
    }

    string_t str1 = string_create("%d", x->id);
    error = map_get(map, str1, (void **)&x_id);
    if (error != NULL)
    {
        free(str1);
        return error;
    }
    free(str1);

    error = graph_tensor_node(y);
    if (error != NULL)
    {
        return error;
    }

    str1 = string_create("%d", y->id);
    error = map_get(map, str1, (void **)&y_id);
    if (error != NULL)
    {
        free(str1);
        return error;
    }
    free(str1);

    error = graph_tensor_node(z);
    if (error != NULL)
    {
        return error;
    }

    str1 = string_create("%d", z->id);
    error = map_get(map, str1, (void **)&z_id);
    if (error != NULL)
    {
        free(str1);
        return error;
    }
    free(str1);

    op_id = graph_operation_node(operation, true);

    // add edges
    error = graph_edge((uint64_t)x_id, (uint64_t)op_id);
    if (error != NULL)
    {
        return error;
    }
 
    error = graph_edge((uint64_t)y_id, (uint64_t)op_id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_edge((uint64_t)op_id, (uint64_t)z_id);
    if (error != NULL)
    {
        return error;
    }

    file = fopen("graph.dot", "w");
    if (file != NULL)
    {
        agwrite(graph, file);
        fclose(file);
    }

    return NULL;
}

nw_error_t *graph_unary_operation(tensor_t *x, tensor_t *y, string_t operation)
{
    uint64_t *x_id = NULL, *y_id = NULL, op_id;
    nw_error_t *error;

    // add nodes
    error = graph_tensor_node(x);
    if (error != NULL)
    {
        return error;
    }

    string_t str1 = string_create("%d", x->id);
    error = map_get(map, str1, (void **)&x_id);
    if (error != NULL)
    {
        free(str1);
        return error;
    }
    free(str1);

    error = graph_tensor_node(y);
    if (error != NULL)
    {
        return error;
    }

    str1 = string_create("%d", y->id);
    error = map_get(map, str1, (void **)&y_id);
    if (error != NULL)
    {
        free(str1);
        return error;
    }
    free(str1);
 
    op_id = graph_operation_node(operation, false);

    // add edges
    error = graph_edge((uint64_t)x_id, (uint64_t)op_id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_edge((uint64_t)op_id, (uint64_t)y_id);
    if (error != NULL)
    {
        return error;
    }

    file = fopen("graph.dot", "w");
    if (file != NULL)
    {
        agwrite(graph, file);
        fclose(file);
    }

    return NULL;
}
