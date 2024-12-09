from idlmav_types import MavNode, MavGraph
from typing import overload

class MavMerger():
    def __init__(self, graph:MavGraph, cumul_param_threshold=0.01):
        self.g = graph
        self.merge_nodes(cumul_param_threshold)

    def merge_nodes(self, cumul_param_threshold):
        """
        `merge_nodes` marks some nodes as top-level and others
        as merged. When drawn, top-level nodes will be placed at 
        integer coordinates and merged nodes the fractional parts 
        between integer coordinates. 
        
        For example, an activation module or function can often be 
        drawn close to the previous module and might not require 
        a row of its own.

        Merging restrictions:
        * Graph input and output nodes cannot be merged
        * Nodes with multiple input connections cannot be merged
        * Nodes for which the input node has multiple output
          connections cannot be merged

        Subject to these restrictions, the default implementation
        sorts the nodes in ascending order of the number of
        parameters and starts merging from the smallest node until 
        the merged nodes cumulatively contribute to a fraction just 
        below `cumul_param_threshold` of the total parameters in the 
        model.

        Setting `cumul_param_threshold` to zero merges only nodes
        with zero paramters. Setting `cumul_param_threshold` to a
        negative value disables merging.
        """
        total_params = sum([n.params for n in self.g.nodes])
        cumul_param_cutoff = total_params * cumul_param_threshold
        sorted_nodes = sorted(self.g.nodes, key=lambda n: n.params)
        cumul_params = 0
        for n in sorted_nodes:
            cumul_params += n.params
            n.is_subnode = can_merge_node(n) and (cumul_params <= cumul_param_cutoff)

        # Ensure that for each operation, some nodes are not merged and others not
        unmerged_operations = set([n.operation for n in self.g.nodes if not n.is_subnode])
        for op in unmerged_operations:
            nodes = [n for n in self.g.nodes if n.operation == op]
            for n in nodes: n.is_subnode = False

        # Update graph and node properties based on sub-node allocation
        self.g.update_top_level_lists()

def can_merge_node(n:MavNode) -> bool:
    if len(n._in_nodes) != 1: return False  # Node is either an input mode or a branch is merged at this node
    if not n._out_nodes: return False  # Node is an output node
    in_node = n._in_nodes[0]
    if len(in_node._out_nodes) != 1: return False  # Node is one level below a branch
    return True
    
@overload
def merge_graph_nodes(graph:MavGraph, 
                      cumul_param_threshold=0.01): ...
def merge_graph_nodes(g:MavGraph, *args, **kwargs):
    merger = MavMerger(g, *args, **kwargs)