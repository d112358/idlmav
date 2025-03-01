from .mavoptions import MavOptions
from .mavtypes import MavNode, MavGraph
from typing import overload

class MavMerger():
    """
    Class that performs the merging step

    All processing is performed upon instantiation.

    This class is not used after this, so users are encouraged to access
    it via `merge_graph_nodes`, except if sub-classed.
    """
    def __init__(self, graph:MavGraph, opts:MavOptions=MavOptions(), **kwargs):
        """
        Instantiates a `MavMerger` class and performs the merging step.

        See `merge_graph_nodes` for a description of arguments
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        self.g = graph
        self.merge_nodes(opts.merge_threshold)

    def merge_nodes(self, cumul_param_threshold):
        """
        Marks certain nodes as top-level and others as merged. When drawn, 
        top-level nodes will be placed at integer coordinates and merged 
        nodes the fractional parts between integer coordinates. 
        
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
            if op.endswith('()'): continue  # Don't apply to function calls
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
    
def merge_graph_nodes(g:MavGraph, opts:MavOptions=MavOptions(), **kwargs):
    """
    Performs the merging step
    
    Keyword arguments may be passed either via a `MavOptions` object or
    as-is. Using a `MavOptions` object provides better intellisense, 
    but plain keyword arguments results in more concise code.

    The following two lines are equivalent:
    ```
    merge_graph_nodes(g, MavOptions(merge_threshold=0.1))  
    merge_graph_nodes(g, merge_threshold=0.1)  
    ```

    Parameters
    ----------
    g: MavGraph
        Graph object produced by tracing step

    merge_threshold: float
        Determines the amount of merging to perform:
        * Negative values disable merging altogether
        * A value of zero causes only nodes without any parameters to be merged
        * For values between 0 and 1, all nodes will be sorted in ascending order 
          of the number of parameters and merged from the smallest node until
          a cumulative fraction of merge_threshold is reached
        * The default value of 0.01 typically causes nodes without parameters
          and very small nodes such as normalization operations to be merged    
    """
    for k,v in kwargs.items(): opts.__setattr__(k,v)
    merger = MavMerger(g, opts)