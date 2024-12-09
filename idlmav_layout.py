from idlmav_types import MavNode, MavConnection, MavGraph
from typing import List, Tuple
import numpy as np
import itertools
from collections import deque
from munkres import Munkres
import random

class MavLayout():
    def __init__(self, graph:MavGraph):
        self.g = graph
        self.calc_layout()

    def calc_layout(self):
        # Determine the level on which each top-level node should go
        # * Connections are only from earlier to later levels, not in the other
        #   direction and not between nodes on the same level
        levels = self.calc_levels()

        # Find the level containing the most top-level nodes
        level_lens = [len(level) for level in levels]
        largest_level_idx = np.argmax(level_lens)
        largest_level_len = level_lens[largest_level_idx]

        # Try all permutations of top-level nodes on the largest level
        # * For each permutation, use the Hungarian method (via munkres library) to
        #   iteratively determine the best placement of top-level nodes on the levels 
        #   before and after this level
        xs = [i-(largest_level_len-1)//2 for i in range(largest_level_len)]
        xs_perms = list(itertools.permutations(xs))
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
            if n in self.g.in_nodes:
                n.y = 0
            else:
                n.y = n.y - min_level_number
            nodes_on_each_level[n.y].append(i)

        return nodes_on_each_level

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

def layout_graph_nodes(g:MavGraph, *args, **kwargs):
    layout = MavLayout(g, *args, **kwargs)

def create_random_sample_graph(nodes_per_level, num_connections, rep_prob_decay=0.1, skip_prob_decay=0.1):
    """
    Creates a sample graph with the approximate structure as specified

    Note:
    * The purpose of this function is to generate sample graphs to test and demonstrate the layout
      and interactive visualization functionality of this library using networks of different shapes
      and sizes. 
      - For this purpose, it was deemed sufficient if this function treats it specified input
        parameters as approximate guidelines. No effort was invested in meeting these parameters
        exactly across a wide range of inputs. That being said, the first two parameters are often
        met exactly for sensible and compatible ranges of values.
    * Metadata (e.g. activations, parameters, FLOPS) generated by this function are enirely fictional
      and will not make sense if used in any mathematical analysis. These serve purely to test and
      demonstrate the interactive visualization functionality of this library

    Parameters:
    * `nodes_per_level` is a listof the number of nodes on each level, e.g. [1,2,3,4,3,2,1]
    * `num_connections` is a suggestion for the total number of connections in the network. The
      actual total number of connections may be more if required to establish the levels as 
      specified
    * `rep_prob_decay` is the fraction by which the probability of a node being chosen as input
       is multiplied each time that node is chosen. Values between 0 and 1 are recommended.
    * `skip_prob_decay` is the fraction by which the probability of a node being chosen as input
      to another node decays as the two nodes move further apart in levels. Values between 0 and
      1 are recommended
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
