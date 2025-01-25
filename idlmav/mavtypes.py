from typing import List
import warnings

class MavNode:
    def __init__(self, name, y, x):
        self.name, self.y, self.x = name, y, x
        self.operation = ''
        self.activations = (0,)
        self.params = 0
        self.flops = 0
        self.is_subnode = False
        self.op_color = '#bf7f7f'  # Faded red, to indicate an error if not assigned, but distinguish from error due to failed forward pass
        self.error = None
        self.metadata = {}
        self._idx = None  # Index within graph
        self._top_level_idx = None  # Index within graph
        self._in_nodes:List[MavNode] = []
        self._out_nodes:List[MavNode] = []
        self._subnodes:List[MavNode] = []
        self._top_level_in_nodes:List[MavNode] = []
        self._top_level_out_nodes:List[MavNode] = []
    def __repr__(self):
        return f'MavNode: name={self.name}; (x,y)=({self.x},{self.y}); in={[n.name for n in self._in_nodes]}, out={[n.name for n in self._out_nodes]}'

class MavConnection:
    def __init__(self, from_node:MavNode, to_node:MavNode):
        self.from_node, self.to_node = from_node, to_node
        self.offset = None  # Used for drawing skip connections around multiple nodes in a straight line

class MavGraph:
    def __init__(self, nodes:List[MavNode], connections:List[MavConnection]):
        self.nodes = nodes
        self.connections = connections
        self.in_nodes:List[MavNode] = []  # Updated in _assign_in_out_nodes
        self.out_nodes:List[MavNode] = []  # Updated in _assign_in_out_nodes
        self.top_level_nodes:List[MavNode] = []  # Updated in update_top_level_lists
        self.top_level_connections:List[MavConnection] = []  # Updated in update_top_level_lists
        self.nw = 0.5  # Node width
        self.nh = 0.5  # Node height
        self._assign_in_out_nodes()
        self.update_top_level_lists()

    def __repr__(self):
        lines = []
        lines.append(f'Inputs: {[n.name for n in self.in_nodes]}')
        lines.append(f'Outputs: {[n.name for n in self.out_nodes]}')
        lines.append('All nodes:')
        lines += [f'    {n.__repr__()}' for n in self.nodes]
        return '\n'.join(lines)

    def _assign_in_out_nodes(self):
        """
        `_assign_in_out_nodes` updates the following fields:
        * `MavNode._in_nodes`
        * `MavNode._out_nodes`
        * `MavGraph.in_nodes`
        * `MavGraph.out_nodes`

        The following fields must be assigned correctly before 
        calling `_assign_in_out_nodes`:
        * `MavGraph.nodes`
        * `MavGraph.connections
        * `MavConnection.to_node`
        * `MavConnection.from_node`
        """
        for i,n in enumerate(self.nodes): n._idx = i

        for c in self.connections:
            if not c.to_node in c.from_node._out_nodes:
                c.from_node._out_nodes.append(c.to_node)
            if not c.from_node in c.to_node._in_nodes:
                c.to_node._in_nodes.append(c.from_node)
        
        # Graph input and output nodes
        self.in_nodes = [node for node in self.nodes if not node._in_nodes]
        self.out_nodes = [node for node in self.nodes if not node._out_nodes]

    def update_top_level_lists(self):
        """
        `update_top_level_lists` updates the following fields:
        * `MavGraph.top_level_nodes`
        * `MavGraph.top_level_connections`
        * `MavNode._top_level_idx`
        * `MavNode._subnodes`
        * `MavNode._top_level_in_nodes`
        * `MavNode._top_level_out_nodes`

        The following fields must be assigned correctly before 
        calling `update_top_level_lists`:
        * `MavGraph.nodes`
        * `MavNode._in_nodes`
        * `MavNode._out_nodes`
        * `MavNode.is_subnode`
        """

        # Clear some lists
        self.top_level_connections = []
        for n in self.nodes:
            n._top_level_in_nodes = []
            n._top_level_out_nodes = []
            n._subnodes = []

        # Identify top-level nodes and walk from each one
        self.top_level_nodes = [n for n in self.nodes if not n.is_subnode]
        for tni, tn in enumerate(self.top_level_nodes):
            tn._top_level_idx = tni
            tn._subnodes = []
            done = False
            while not done:
                cur_node = tn._subnodes[-1] if tn._subnodes else tn
                # 4 Possibilities:
                # * Current node has no outputs: we are done with this top-level node
                # * Current node has 1 subnode output: continue walking
                # * Current node has 1 top-level output: we are done with this top-level node
                # * Current node has multiple outputs: they must all be top-level by definition and we are done

                still_busy = False  # Rather indicate explicitly that we want to continue the loop
                                    # This is safer in terms of accidental infinite loops after
                                    # editing this later
                if not cur_node._out_nodes:
                    # Current node has no outputs
                    # * Warn if it is not an output node and branch just stops
                    if tn._out_nodes:  # Don't warn if top-level node is an output node
                        warnings.warn(f'Path from top-level node {tn.name} does not reach another top-level node')
                elif len(cur_node._out_nodes) == 1 and cur_node._out_nodes[0].is_subnode:
                    # Current node has 1 subnode output: continue walking
                    out_node = cur_node._out_nodes[0]
                    out_node._top_level_in_nodes = [tn]
                    tn._subnodes.append(out_node)
                    still_busy = True
                else:
                    # Current node has 1 or more top-level outputs
                    # * Warn if branch has any subnodes
                    for out_node in cur_node._out_nodes:
                        if out_node.is_subnode:
                            warnings.warn(f'Branch from {cur_node.name} includes subnode {out_node.name}')
                        else:
                            tc = MavConnection(tn, out_node)
                            self.top_level_connections.append(tc)
                            tn._top_level_out_nodes.append(out_node)
                            out_node._top_level_in_nodes.append(tn)
                            for sn in tn._subnodes:
                                # sn._top_level_in_nodes are updated above
                                sn._top_level_out_nodes.append(out_node)
                done = not still_busy