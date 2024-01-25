/**@file graph.h
 * @brief Provides graph node type and its utilities.
 *
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <datatype.h>
#include <tensor.h>
#include <map.h>
#include <function.h>
#include <buffer.h>
#include <view.h>
#include <element.h>
#include <stdarg.h>
#include <graphviz/cgraph.h>

nw_error_t *start_graph(void);
void end_graph(void); 
nw_error_t *graph_function(function_t *function, tensor_t *result);

#endif