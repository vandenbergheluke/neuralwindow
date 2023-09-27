/**@file graph.h
 * @brief Provides graph node type and its utilities.
 *
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <datatype.h>
#include <tensor.h>
#include <stack.h>
#include <buffer.h>
#include <view.h>
#include <element.h>
#include <graphviz/cgraph.h>

typedef struct graph_node_t
{
    uint64_t tensor_id;
    uint64_t graph_id;
} graph_node_t;

typedef enum node_type
{
    OPERATION,
    TENSOR
} node_type;

nw_error_t *initialize_graph();
void destroy_graph();
uint64_t get_graph_id(uint64_t tensor_id);
nw_error_t *create_graph_node(graph_node_t **graph_node, uint64_t tensor_id, uint64_t new_graph_id);
nw_error_t *update_graph_id(uint64_t tensor_id, uint64_t new_graph_id);
string_t get_attribute_format(uint64_t rank, uint64_t *attr);
nw_error_t *graph_tensor_node(tensor_t *tensor);
uint64_t graph_operation_node(string_t op, bool_t binary);
nw_error_t *graph_edge(uint64_t node_1_ID, uint64_t node_2_ID);
nw_error_t *graph_binary_operation(tensor_t *x, tensor_t *y, tensor_t *z, string_t operation);
nw_error_t *graph_unary_operation(tensor_t *x, tensor_t *y, string_t operation);

#endif