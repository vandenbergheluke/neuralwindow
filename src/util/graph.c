#include <graph.h>

static Agraph_t *graph = NULL;
static stack_t *stack = NULL;
static uint64_t node_id = 0;
FILE *file;

nw_error_t *initialize_graph()
{
    graph = agopen("G", Agstrictdirected, 0);
    if (graph == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, "Graph file could not be created", NULL);
    }

    agattr(graph, AGRAPH, "rankdir", "LR");
    agattr(graph, AGNODE, "shape", "record");

    nw_error_t *error = stack_create(&stack);
    if (error != NULL) 
    {
        agclose(graph);
        return ERROR(ERROR_MEMORY_ALLOCATION, "Stack could not be created", NULL);
    }

    return NULL;
}

void destroy_graph()
{
    FILE *fp = fopen("graph.dot", "w");
    agwrite(graph, fp);
    fclose(fp);
    agclose(graph);
    stack_destroy(stack);
}

uint64_t get_graph_id(uint64_t tensor_id)
{
    element_t *current = stack->head;
    while (current != NULL)
    {
        graph_node_t *node = (graph_node_t *)current->data;
        if (node->tensor_id == tensor_id)
        {
            return (uint64_t)node->graph_id;
        }
        current = current->next;
    }
    return (uint64_t)-1;
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

nw_error_t *update_graph_id(uint64_t tensor_id, uint64_t new_graph_id)
{
    CHECK_NULL_ARGUMENT(stack, "stack");

    element_t *current = stack->head;
    while (current != NULL)
    {
        graph_node_t *node = (graph_node_t *)current->data;
        if (node->tensor_id == tensor_id)
        {
            node->graph_id = new_graph_id;
            return NULL;
        }
        current = current->next;
    }

    // graph node not found, create new node in stack
    graph_node_t *graph_node;
    nw_error_t *error = create_graph_node(&graph_node, tensor_id, new_graph_id);
    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("Could not create graph node for tensor %lu.", tensor_id), NULL);
    }

    error = stack_push(stack, graph_node);
    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("Could not add graph node on stack for tensor %lu.", tensor_id), NULL);
    }

    return NULL;
}

string_t get_attribute_format(uint64_t rank, uint64_t *attr)
{
    switch(rank)
    {
        case 1:
            return string_create("(%d)", *attr);
        case 2:
            return string_create("(%d, %d)", *attr, *(attr + 1));
        case 3:
            return string_create("(%d, %d, %d)", *attr, *(attr + 1), *(attr + 2));
        case 4:
            return string_create("(%d, %d, %d, %d)", *attr, *(attr + 1), *(attr + 2), *(attr + 3));
        default:
            return NULL;
    }
}

nw_error_t *graph_tensor_node(tensor_t *tensor)
{   
    if (graph == NULL || stack == NULL)
    {
        
        nw_error_t *error = initialize_graph();
        if (error != NULL)
        {
            return error;
        }
        
        error = NULL;
    }


    uint64_t rank = tensor->buffer->view->rank;
    string_t node_label = string_create("<F0> Tensor_ID: %d|Shape: %s|Size: %d|Stride: %s|Offset: %d",  tensor->id,
                                                                                  get_attribute_format(rank, tensor->buffer->view->shape),
                                                                                  tensor->buffer->storage->n,
                                                                                  get_attribute_format(rank, tensor->buffer->view->strides),
                                                                                  tensor->buffer->view->offset);
    
    // check if there is already a node associated with the tensor ID on the graph
    uint64_t prev_node_id, new_node_it;
    prev_node_id = get_graph_id(tensor->id);
    if (prev_node_id == (uint64_t)-1)
    {
        new_node_it = ++node_id;
    }
    else
    {
        new_node_it = prev_node_id;
    }

    // update graph                                                                        
    Agnode_t *node = agnode(graph, (char *)string_create("%d", new_node_it), 1);
    if (node == NULL) 
    {
        return ERROR(ERROR_GRAPH, string_create("Could not add tensor node on the graph for tensor: %d", tensor->id), NULL); 
    }
    agsafeset(node, "label", (char *)node_label, "");

    // update stack
    nw_error_t *error = update_graph_id(tensor->id, new_node_it);    
    return error;
}

uint64_t graph_operation_node(string_t op, bool_t binary)
{   
    // update graph                                                                        
    Agnode_t *node = agnode(graph, (char *)string_create("%d", ++node_id), 1);
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
    return node_id;
}

nw_error_t *graph_edge(uint64_t node_1_ID, uint64_t node_2_ID)
{
    Agnode_t *node1, *node2;
    node1 = agnode(graph, (char *)string_create("%d", node_1_ID), 0);
    node2 = agnode(graph, (char *)string_create("%d", node_2_ID), 0);

    Agedge_t *edge = agedge(graph, node1, node2, NULL, 1);
    if (edge == NULL) 
    {
        return ERROR(ERROR_GRAPH, "Could not add edge on the graph", NULL); 
    }
    return NULL;
}


nw_error_t *graph_binary_operation(tensor_t *x, tensor_t *y, tensor_t *z, string_t operation)
{
    uint64_t x_id, y_id, z_id, op_id;
    nw_error_t *error;

    // add nodes
    error = graph_tensor_node(x);
    x_id = get_graph_id(x->id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_tensor_node(y);
    y_id = get_graph_id(y->id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_tensor_node(z);
    z_id = get_graph_id(z->id);
    if (error != NULL)
    {
        return error;
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
    uint64_t x_id, y_id, op_id;
    nw_error_t *error;
    
    // add nodes
    error = graph_tensor_node(x);
    x_id = get_graph_id(x->id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_tensor_node(y);
    y_id = get_graph_id(y->id);
    if (error != NULL)
    {
        return error;
    }

    op_id = graph_operation_node(operation, false);

    // add edges
    error = graph_edge(x_id, op_id);
    if (error != NULL)
    {
        return error;
    }

    error = graph_edge(op_id, y_id);
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
