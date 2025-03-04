from .mavoptions import MavOptions
from .mavtypes import MavNode, MavConnection, MavGraph
from typing import List, Tuple, Set, Mapping
import warnings
from numpy.typing import NDArray
import math
import numpy as np
import itertools
from collections import deque
from munkres import Munkres
import random

class MavLayout():
    """
    Class that performs the layout step

    All processing is performed upon instantiation.

    This class is not used after this, so users are encouraged to access
    it via `layout_graph_nodes`, except if sub-classed.
    """
    def __init__(self, graph:MavGraph, opts:MavOptions=MavOptions(), **kwargs):
        """
        Instantiates a `MavLayout` class and performs the layout step.

        See `layout_graph_nodes` for a description of arguments
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        self.g = graph
        self.calc_layout()

    def calc_layout(self):
        # Determine the level on which each top-level node should go
        # * Connections are only from earlier to later levels, not in the other
        #   direction and not between nodes on the same level
        levels = self.calc_levels()

        # Determine the placement of nodes within each level
        self.place_within_levels(levels)

    def calc_levels(self, verbose=False):
        
        # Breadth-first search
        queue = deque(self.g.out_nodes)  # Initialize to contain all output nodes
        visited = set(self.g.out_nodes)
        min_level_number = 0
        while queue:
            if verbose: print([n.name for n in queue])
            cur_node = queue.popleft()
            if cur_node._top_level_out_nodes:
                # Set level to just before than earliest output node
                cur_node.y = min([n.y-1 for n in cur_node._top_level_out_nodes])
            else:
                # Node is an output node. Set level to 0
                cur_node.y = 0

            if cur_node.y < min_level_number: min_level_number = cur_node.y

            # Queue all input nodes of current node that have all their outputs visited
            for in_node in cur_node._top_level_in_nodes:
                if in_node in visited: continue
                if not all([o in visited for o in in_node._top_level_out_nodes]): continue
                queue.append(in_node)
                visited.add(in_node)

        # Adjust to have inputs at level 0
        # * Other nodes move up by the same amount
        # * Also populate the return object: A list of lists of nodes indices at each level
        num_levels = 1 - min_level_number
        nodes_on_each_level = [[] for i in range(num_levels)]
        for i,n in enumerate(self.g.top_level_nodes):
            if n in self.g.in_nodes and n.metadata.get('entry_type','normal') == 'normal':
                n.y = 0
            else:
                n.y = n.y - min_level_number
            nodes_on_each_level[n.y].append(i)

        return nodes_on_each_level
    
    def place_within_levels(self, levels):
        # Choose an algorithm based on the number of nodes in the largest level
        level_lens = [len(level) for level in levels]
        largest_level_len = max(level_lens)
        if largest_level_len > 5:
            print(f'Total nodes: {len(self.g.nodes)}. Input nodes: {len(self.g.in_nodes)}. Output nodes: {len(self.g.out_nodes)}. Largest level nodes: {largest_level_len}')
        if largest_level_len > 15:
            warnings.warn(f'The largest level has {largest_level_len} nodes. This may be an indication that something may have gone wrong during the tracing step')
            warnings.warn('Falling back to greedy layout algorithm')
            self.place_within_levels_greedy(levels)
        else:
            self.place_within_levels_munkres(levels)

    def place_within_levels_greedy(self, levels):
        # Reverse breadth-first search to determine total number of top-level nodes accessible from each top-level node
        queue = deque(self.g.out_nodes)  # Initialize to contain all output nodes
        visited:Set[MavNode] = set([])
        cumul_num_outputs:Mapping[MavNode, int] = {}
        while queue:
            cur_node = queue.popleft()
            visited.add(cur_node)

            # Ensure that cumulative_num_outputs is defined for current node
            if cur_node not in cumul_num_outputs: cumul_num_outputs[cur_node] = 0

            # Update cumulative_num_outputs for all input nodes
            for in_node in cur_node._top_level_in_nodes:
                if in_node in cumul_num_outputs:
                    cumul_num_outputs[in_node] += cumul_num_outputs[cur_node] + 1
                else:
                    cumul_num_outputs[in_node] = cumul_num_outputs[cur_node] + 1
                    
            # Queue all input nodes of current node that have all their outputs visited
            for in_node in cur_node._top_level_in_nodes:
                if in_node in visited: continue
                if not all([o in visited for o in in_node._top_level_out_nodes]): continue
                queue.append(in_node)

        # At each level, place top-level nodes in order of total number of accessible output nodes
        wc, wd = 1,10
        level_lens = [len(level) for level in levels]
        largest_level_len = max(level_lens)
        candidate_xs = [i-(largest_level_len-1)//2 for i in range(largest_level_len)]
        xs = [None] * len(self.g.top_level_nodes)
        for cur_level in levels:
            cur_level_nodes = [self.g.top_level_nodes[ni] for ni in cur_level]
            cur_level_scores = [cumul_num_outputs[n] for n in cur_level_nodes]
            sorted_nodes:List[MavNode] = [n for n,_ in sorted(zip(cur_level_nodes, cur_level_scores), key=lambda x: x[1], reverse=True)]
            used_xs:Set[int] = set([])
            for ni,n in enumerate(sorted_nodes):
                available_xs = [x for x in candidate_xs if x not in used_xs]
                cost_vector = np.zeros((len(available_xs),))  # Cost of placing node at each candidate x-coordinate
                for xi,x in enumerate(available_xs):
                    cc = abs(x)  # Cost of placing node away from the center
                    cd = 0       # Cost of placing node away from connected input node
                    for in_node in n._top_level_in_nodes:
                        if xs[in_node._top_level_idx] is not None:
                            cd += abs(x - xs[in_node._top_level_idx])
                    cost_vector[xi] = wc*cc + wd*cd
                chosen_xi = np.argmin(cost_vector)
                n.x = available_xs[chosen_xi]
                xs[n._top_level_idx] = n.x
                used_xs.add(n.x)

        # Place subnodes below the corresponding top-level nodes
        for tn in self.g.top_level_nodes:
            num_subnodes = len(tn._subnodes)
            subnode_interval = 0.15 if num_subnodes <= 4 else 0.6/num_subnodes
            for sni, sn in enumerate(tn._subnodes):
                sn.x = tn.x
                sn.y = tn.y + (sni+1)*subnode_interval

    def place_within_levels_munkres(self, levels):
        # Find the level containing the most top-level nodes
        level_lens = [len(level) for level in levels]
        largest_level_idx = np.argmax(level_lens)
        largest_level_len = level_lens[largest_level_idx]

        # Try all permutations of top-level nodes on the largest level
        # * For each permutation, use the Hungarian method (via munkres library) to
        #   iteratively determine the best placement of top-level nodes on the levels 
        #   before and after this level
        xs_perms = self.sample_permutations(largest_level_len)
        num_xs_perms = len(xs_perms)
        total_costs = [None] * num_xs_perms
        xdata = np.zeros((num_xs_perms, len(self.g.top_level_nodes)))
        for prmi, xs_perm in enumerate(xs_perms):
            xdata[prmi,:], total_costs[prmi] = self.best_layout_one_fixed_level(levels, largest_level_idx, xs_perm)

        # Select the permutation with the smallest total cost
        best_idx = np.argmin(total_costs)
        best_xs = xdata[best_idx]
        for ni,x in enumerate(best_xs):
            self.g.top_level_nodes[ni].x = x

        # Place subnodes below the corresponding top-level nodes
        for tn in self.g.top_level_nodes:
            num_subnodes = len(tn._subnodes)
            subnode_interval = 0.15 if num_subnodes <= 4 else 0.6/num_subnodes
            for sni, sn in enumerate(tn._subnodes):
                sn.x = tn.x
                sn.y = tn.y + (sni+1)*subnode_interval


    def sample_permutations(self, largest_level_len):
        max_num_permutations = 1000
        xs = [i-(largest_level_len-1)//2 for i in range(largest_level_len)]
        total_permutations = math.factorial(largest_level_len)
        if total_permutations < max_num_permutations:
            xs_perms = list(itertools.permutations(xs))
        else:
            rng = np.random.default_rng()
            xs_perms = [list(rng.permutation(xs)) for _ in range(max_num_permutations)]
        return xs_perms

    def best_layout_one_fixed_level(self, levels:List[List[int]], fixed_level, fixed_xs:List[int], wc=1, wd=10, verbose=False):
        level_lens = [len(level) for level in levels]
        largest_level_len = max(level_lens)
        candidate_xs = [i-(largest_level_len-1)//2 for i in range(largest_level_len)]
        xs = [None] * len(self.g.top_level_nodes)
        total_cost = 0
        M = Munkres()

        # Fixed level
        for xi, ni in enumerate(levels[fixed_level]): 
            xs[self.g.top_level_nodes[ni]._top_level_idx] = fixed_xs[xi]
            total_cost += wc*abs(fixed_xs[xi])

        # Levels before fixed level
        if fixed_level > 0:
            for cur_level in levels[fixed_level-1::-1]:
                cur_level_len = len(cur_level)
                cur_level_nodes = [self.g.top_level_nodes[ni] for ni in cur_level]
                cost_matrix = np.zeros((cur_level_len, largest_level_len))  # Cost of placing each cur_level node at each candidate x-coordinate
                for ni,n in enumerate(cur_level_nodes):
                    for xi,x in enumerate(candidate_xs):
                        cc = abs(x)  # Cost of placing node away from the center
                        cd = 0       # Cost of placing node away from connected output node
                        for out_node in n._top_level_out_nodes:
                            if xs[out_node._top_level_idx] is not None:
                                cd += abs(x - xs[out_node._top_level_idx])
                            else:
                                pass  # Skip connection between levels before and after fixed_level. Will be counted in other direction.
                        cost_matrix[ni,xi] = wc*cc + wd*cd
                best_path = M.compute(cost_matrix.copy())
                level_cost = 0
                for ni, xi in best_path:
                    xs[cur_level_nodes[ni]._top_level_idx] = candidate_xs[xi]
                    level_cost += cost_matrix[ni,xi]
                total_cost += level_cost
                if verbose:
                    print(f'cost_matrix={cost_matrix}')
                    print(f'best_path={best_path}')
                    print(f'xs={xs}')
                    print('')

        # Levels after fixed level
        if fixed_level < len(levels) -1:
            for cur_level in levels[fixed_level+1:]:
                cur_level_len = len(cur_level)
                cur_level_nodes = [self.g.top_level_nodes[ni] for ni in cur_level]
                cost_matrix = np.zeros((cur_level_len, largest_level_len))  # Cost of placing each cur_level node at each candidate x-coordinate
                for ni,n in enumerate(cur_level_nodes):
                    for xi,x in enumerate(candidate_xs):
                        cc = abs(x)  # Cost of placing node away from the center
                        cd = 0       # Cost of placing node away from connected input node
                        for in_node in n._top_level_in_nodes:
                            if xs[in_node._top_level_idx] is not None:
                                cd += abs(x - xs[in_node._top_level_idx])
                            else:
                                pass  # Skip connection between levels before and after fixed_level. Will be counted in other direction.
                        cost_matrix[ni,xi] = wc*cc + wd*cd
                best_path = M.compute(cost_matrix.copy())
                level_cost = 0
                for ni, xi in best_path:
                    xs[cur_level_nodes[ni]._top_level_idx] = candidate_xs[xi]
                    level_cost += cost_matrix[ni,xi]
                total_cost += level_cost
                if verbose:
                    print(f'cost_matrix={cost_matrix}')
                    print(f'best_path={best_path}')
                    print(f'xs={xs}')
                    print('')
        return xs, total_cost
    
class SkipConnectionOffsetCalc():
    """
    Calculate appropriate offsets for skip connections between multiple nodes in a straight line.
    See [13_layout_skip_connections.ipynb](../nbs/13_layout_skip_connections.ipynb) for a
    description of the algorithm
    """
    def __init__(self, graph:MavGraph, opts:MavOptions=MavOptions(), **kwargs):
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        self.g = graph
        self.levels:Mapping[float,int] = {}  # Map from the connecion y-value to the level index
        self.skip_connections:List[MavConnection] = []  # List of skip connections in column currently being processed
        self.A:NDArray = None   # Potential overlap matrix. A[i,j] is the number of intersections if connection j is placed outside connection i
        self.B:NDArray = None   # Copy of Potential overlap matrix gradually set to zero as connections are placed
        self.C:NDArray = None   # Collision avoidance matrix. C[i,j] is non-zero if a connection already passes the i'th level in the j'th column
        self.calc_offsets()
        
    def calc_offsets(self):
        xs = list(set([n.x for n in self.g.nodes]))
        for x in xs: self.calc_offsets_for_column(x)

    def calc_offsets_for_column(self, x:int):
        self.init_level_mappings(x)
        self.skip_connections = [c for c in self.g.connections if c.from_node.x == x and c.to_node.x == x and self.is_skip_connection(c)]
        if not self.skip_connections: return
        self.step1_calc_potential_overlap_matrix()
        left_idxs, right_idxs = self.step2_assign_connections_to_sides()
        self.step3_calc_offsets(left_idxs, right_idxs)

    def init_level_mappings(self, x:int):
        ys = [n.y for n in self.g.nodes if n.x == x]
        ys = sorted(list(set(ys)))  # Extract unique values and sort
        self.levels = {y:yi for yi,y in enumerate(ys)}

    def is_skip_connection(self, c:MavConnection):
        n0, n1 = c.from_node, c.to_node
        x0, y0, x1, y1 = n0.x, n0.y, n1.x, n1.y
        if x0 != x1: return False 
        nodes_on_line = [n for n in self.g.nodes if n.x == x0]  # Perform 1st check on all nodes
        nodes_on_segment = [n for n in nodes_on_line if n.y > y0 and n.y < y1]  # Perform 2nd and 3rd checks on subset of nodes
        return True if nodes_on_segment else False

    def step1_calc_potential_overlap_matrix(self):
        # Step 1 in 13_layout_skip_connections.ipynb
        num_skip_connections = len(self.skip_connections)
        self.A = np.zeros((num_skip_connections, num_skip_connections))
        for i0,c0 in enumerate(self.skip_connections):
            y0_from, y0_to = c0.from_node.y, c0.to_node.y
            for i1,c1 in enumerate(self.skip_connections):
                if i1==i0: continue
                y1_from, y1_to = c1.from_node.y, c1.to_node.y
                if y1_from > y0_from and y1_from < y0_to: self.A[i0,i1] += 1
                if y1_to > y0_from and y1_to < y0_to: self.A[i0,i1] += 1
    
    def step2_assign_connections_to_sides(self) -> Tuple[List[int], List[int]]:
        # Step 2 in 13_layout_skip_connections.ipynb
        num_skip_connections = len(self.skip_connections)
        A1 = self.A.copy()
        A1[A1==2] = 0  # Consider only 1's
        AL = np.zeros(self.A.shape)
        AR = np.zeros(self.A.shape)
        left_idxs:List[int] = []
        right_idxs:List[int] = []
        unassigned_idxs = set([i for i in range(num_skip_connections)])
        
        col_sums = A1.sum(axis=0)
        idx = col_sums.argmax()
        self.step2_assign(idx, False, A1, AL, AR, left_idxs, right_idxs, unassigned_idxs)
        while unassigned_idxs:
            cost_diffs = AR.sum(axis=0) - AL.sum(axis=0)
            abs_cost_diffs:NDArray = np.abs(cost_diffs)
            abs_cost_diffs[left_idxs + right_idxs] = -1
            idx = abs_cost_diffs.argmax()  # TODO: tie-break on smallest total column sum in A??
            to_right = cost_diffs[idx] <= 0
            self.step2_assign(idx, to_right, A1, AL, AR, left_idxs, right_idxs, unassigned_idxs)

        for idx in left_idxs: self.skip_connections[idx].offset = -1
        for idx in right_idxs: self.skip_connections[idx].offset = 1

        return left_idxs, right_idxs
    
    def step2_assign(self, idx:int, to_right:bool, A1:NDArray, AL:NDArray, AR:NDArray, left_idxs:List[int], right_idxs:List[int], unassigned_idxs:Set[int]):
        if to_right:
            AR[idx,:] = A1[idx,:]
            left_idxs.append(idx)
        else:
            AL[idx,:] = A1[idx,:]
            right_idxs.append(idx)
        unassigned_idxs.remove(idx)

    def step3_calc_offsets(self, left_idxs:List[int], right_idxs:List[int]):
        # Step 3 in 13_layout_skip_connections.ipynb
        for side in range(2):
            if side==0:
                side_connections = [self.skip_connections[i] for i in left_idxs]
                self.B = self.A.copy()[left_idxs,:][:,left_idxs]
                side_factor = -1
            else:
                side_connections = [self.skip_connections[i] for i in right_idxs]
                self.B = self.A.copy()[right_idxs,:][:,right_idxs]
                side_factor = 1
            if not side_connections: continue
            num_side_connections = len(side_connections)
            max_level = max([self.levels[c.to_node.y] for c in side_connections])
            self.C = np.zeros((max_level+1,1))

            placed_idxs:List[int] = []
            unplaced_idxs = set([i for i in range(num_side_connections)])
            while unplaced_idxs:
                idx = self.step3_get_idx_to_place(placed_idxs)
                self.step3_place(placed_idxs, unplaced_idxs, idx, side_connections[idx], side_factor)

        # Normalize
        max_abs_offset = max([abs(c.offset) for c in self.skip_connections])
        scale_factor = 0.4 / max_abs_offset
        for c in self.skip_connections: c.offset = c.offset*scale_factor

    def step3_ensure_C_width(self, w:int):
        if self.C.shape[1] >= w: return
        self.C = np.append(self.C, np.zeros((self.C.shape[0], w-self.C.shape[1])), axis=1)

    def step3_get_idx_to_place(self, placed_idxs:List[int]):
        # Check for any unplaced index with a zero row sum
        row_sums = self.B.sum(axis=1)
        row_sums[placed_idxs] = 999999
        zero_row_sum_idxs = (row_sums==0).nonzero()[0]
        if len(zero_row_sum_idxs) > 0: return zero_row_sum_idxs[0]

        # Check for any index with 2's in columns, but not in rows
        # * Placed indices are already zeroed out
        col_2_counts = (self.B==2).sum(axis=0)
        row_2_counts = (self.B==2).sum(axis=0)
        diff_2_counts = col_2_counts - row_2_counts
        diff_2_counts[row_2_counts > 0] = 0
        diff_2_counts[placed_idxs] = 0
        if (diff_2_counts>0).any(): return diff_2_counts.argmax()

        # If this point is reached, place the unplaced connection with the lowest row sum
        return row_sums.argmin()

    def step3_place(self, placed_idxs:List[int], unplaced_idxs:Set[int], idx:int, c:MavConnection, side_factor):
        y0, y1 = self.levels[c.from_node.y], self.levels[c.to_node.y]
        x = 0  # Offset at which to place connection
        self.step3_ensure_C_width(x+1)
        occupied = self.C[y0:y1,:].any()
        while occupied:
            x += 1
            self.step3_ensure_C_width(x+1)
            occupied = self.C[y0:y1,x].any()
        self.B[idx,:] = 0
        self.B[:,idx] = 0
        self.C[y0:y1,x] = 1
        c.offset = side_factor * (x+1)  # side_factor if -1 for left or +1 for right hand side
        placed_idxs.append(idx)
        unplaced_idxs.remove(idx)

def layout_graph_nodes(g:MavGraph, opts:MavOptions=MavOptions(), **kwargs):
    """
    Performs the layout step
    
    Keyword arguments may be passed either via a `MavOptions` object or
    as-is. Using a `MavOptions` object provides better intellisense, 
    but plain keyword arguments results in more concise code.

    The following two lines are equivalent:
    ```
    layout_graph_nodes(g, MavOptions(param_name=param_value))  
    layout_graph_nodes(g, param_name=param_value)  
    ```

    Parameters
    ----------
    g: MavGraph
        Graph object produced by tracing step

    opts: MavOptions and/or keyword arguments
        This step does not currently use any keyword arguments
    """
    for k,v in kwargs.items(): opts.__setattr__(k,v)
    layout = MavLayout(g, opts)
    skip_calc = SkipConnectionOffsetCalc(g, opts)

def create_random_sample_graph(nodes_per_level, num_connections, rep_prob_decay=0.1, skip_prob_decay=0.1):
    """
    Creates a sample graph with the approximate structure as specified

    The purpose of this function is to generate sample graphs to test and demonstrate the layout
    and interactive visualization functionality of this library using networks of different shapes
    and sizes. 

    For this purpose, it was deemed sufficient if this function treats it specified input
    parameters as approximate guidelines. No effort was invested in meeting these parameters
    exactly across a wide range of inputs. That being said, the first two parameters are often
    met exactly for sensible and compatible ranges of values.
    
    Metadata (e.g. activations, parameters, FLOPS) generated by this function are enirely fictional
    and will not make sense if used in any mathematical analysis. These serve purely to test and
    demonstrate the interactive visualization functionality of this library

    Parameters
    ----------
    nodes_per_level: list[int]
        The number of nodes on each level, e.g. `[1,2,3,4,3,2,1]`

    num_connections: int
        A hint for the total number of connections in the network. The actual total 
        number of connections may be more if required to establish the levels as specified.

    rep_prob_decay: float
        The fraction by which the probability of a node being chosen as input is multiplied 
        each time that node is chosen. Values between 0 and 1 are recommended.

    skip_prob_decay: float
        The fraction by which the probability of a node being chosen as input to another node 
        decays as the two nodes move further apart in levels. Values between 0 and 1 are recommended.
    """
    # Create nodes
    num_levels = len(nodes_per_level)
    num_nodes = sum(nodes_per_level)
    levels = []
    for i,num in enumerate(nodes_per_level):
        levels += [i]*num
    nodes = [MavNode(str(ni), 0, 0) for ni in range(num_nodes)]

    # Create main input connection for each node
    p0 = np.array([skip_prob_decay**(num_levels-lvl) for lvl in levels])  # Unscaled probability of picking each node as input
    connection_tuples: List[Tuple[int]] = []
    connections: List[MavConnection] = []
    for n2,node in enumerate(nodes):
        level = levels[n2]
        if level==0: continue  # Input node has no inputs
        p1 = np.where(np.array(levels)<level, p0, 0)
        p = p1 / np.sum(p1)
        n1 = np.random.choice(list(range(len(p))), p=p)
        connection_tuples.append((n1,n2))
        connections.append(MavConnection(nodes[n1], nodes[n2]))
        p0[n1] *= rep_prob_decay

    # Create main output connection for each node
    p0 = np.array([skip_prob_decay**lvl for lvl in levels])  # Unscaled probability of picking each node as output
    for n1,node in enumerate(nodes):
        level = levels[n1]
        if level==num_levels-1: continue  # Output node has no outputs
        if [tpl[1] for tpl in connection_tuples if tpl[0]==n1]: continue  # Node already has outputs
        p1 = np.where(np.array(levels)>level, p0, 0)
        p = p1 / np.sum(p1)
        n2 = np.random.choice(list(range(len(p))), p=p)
        connection_tuples.append((n1,n2))
        connections.append(MavConnection(nodes[n1], nodes[n2]))
        p0[n2] *= rep_prob_decay

    # Create additional connections
    num_attempts = 0
    while len(connections) < num_connections and num_attempts < num_connections*10:
        n1 = np.random.randint(0, num_nodes-1)
        n2 = np.random.randint(0, num_nodes-1)
        if n1>n2: n_temp = n1; n1 = n2; n2 = n_temp
        reject = n1==n2 or (n1,n2) in connection_tuples or levels[n1] >= levels[n2]
        if not reject:
            connection_tuples.append((n1,n2))
            connections.append(MavConnection(nodes[n1], nodes[n2]))
        num_attempts += 1

    # Add fictional metadata
    for n in nodes:
        n.operation = 'sample'
        n.activations = np.random.randint(low=1, high=100, size=(3,)) * 10
        n.params = np.random.randint(low=100, high=1000) * 10
        n.flops = np.random.randint(low=100, high=1000) * 10

    return nodes, connections
