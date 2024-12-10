from .mavtypes import MavNode, MavConnection, MavGraph
from typing import Any, Dict, List, Tuple, Union, Set
import warnings
import torch
from torch import nn, fx, Tensor
import torch.nn.functional as F
from tabulate import tabulate
import torchprofile

class ShapeMacInterpreter(fx.Interpreter):
    def __init__(self, gm : fx.GraphModule):
        super().__init__(gm)

        # Outputs
        self.shapes : Dict[fx.Node, Tuple[int]] = {}
        self.macs : Dict[fx.Node, int] = {}

        # State
        self.running_node = None
        self.last_successful_node = None
        self.cur_macs: int = None

    def run_node(self, n:fx.Node) -> Any:
        # Run the node
        self.cur_macs = None
        self.running_node = n
        result = super().run_node(n)
        self.running_node = None

        # Retrieve the shape
        if isinstance(result, Tensor):
            shape = tuple(result.shape)
        else:
            shape = (0,0,0,0)
        self.shapes[n] = shape

        # Store the number of MACs if calculated
        if n.op == 'call_module':
            if self.cur_macs is not None: self.macs[n] = self.cur_macs

        # Update the state and return the result
        self.last_successful_node = n
        return result
    
    def call_module(self, target, args, kwargs):
        # Run the module
        result = super().call_module(target, args, kwargs)

        # Estimate the FLOPS
        try:
            submod = self.fetch_attr(target)
            macs = torchprofile.profile_macs(submod, args)
        except Exception as e:
            warnings.warn(f'FLOPS calculation failed for module {submod.__class__.__name__}: {e}')
            macs = 0  
            # TODO: implement a fallback calculation here for well-known modules
            # e.g. https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc
        self.cur_macs = macs

        # Return the result
        return result
        
class MavTracer:
    def __init__(self, model:nn.Module, inputs:Union[Tensor, Tuple[Tensor]], device=None):
        if device:
            self.model = model.to(device)
            if isinstance(inputs, Tensor):
                self.inputs = inputs.to(device)
            else:
                self.inputs = tuple(i.to(device) for i in inputs)
        else:
            self.model = model
            self.inputs = inputs
        self.gm: fx.GraphModule = None
        self.interp: ShapeMacInterpreter = None        
        self.g: MavGraph = None
        self.param_counts : Dict[fx.Node, int] = {}
        self.target_names : Dict[fx.Node,str] = {}
        self.err_node: fx.Node = None
        self.run()
        self.build_graph()

    def run(self):
        # 1st pass: symbolic tracing using torch.fx
        self.gm = fx.symbolic_trace(self.model)
        self.interp = ShapeMacInterpreter(self.gm)

        # 2nd pass: iterate through `nn.Module` and update module types and parameter counts
        try:
            for n in self.gm.graph.nodes:
                if n.op == 'call_module':
                    m:nn.Module = self.interp.fetch_attr(n.target)
                    self.target_names[n] = m.__class__.__name__
                    self.param_counts[n] = get_num_trainable_params(m)
                elif n.op == 'call_function':
                    self.target_names[n] = n.target.__name__
        except Exception as e:
            self.err_node = n
            warnings.warn(f'2nd tracing pass failed for module {n.target}: {e}')

        # 3rd pass: forward pass using torch.fx.Interpreter
        try:
            self.interp.run(self.inputs)
        except Exception as e:
            msg = 'Forward pass failed.'
            n1 = self.interp.last_successful_node
            if n1:
                target_name = self.target_names.get(n1, None)
                node_name = f'{n1.name}:{target_name}' if target_name else n1.name
                msg += f' Last successful node: "{node_name}".'
            n2 = self.interp.running_node
            if n2:
                target_name = self.target_names.get(n2, None)
                node_name = f'{n2.name}:{target_name}' if target_name else n2.name
                msg += f' Possible error node: "{node_name}".'
            self.err_node = self.interp.running_node
            warnings.warn(f'{msg}: {e}')

    def get_operation(self, op, target_name):
        match op:
            case 'placeholder': return 'input'
            case 'output': return 'output'
            case 'call_module': return f'nn.{target_name}'
            case 'call_function': return f'{target_name}()'
            case _: return target_name
        
    def build_graph(self):
        nodes: List[MavNode] = []
        nodes_by_name: Dict[str, MavNode] = {}
        connections: List[MavConnection] = []
        existing_connections: Set[Tuple[MavNode, MavNode]] = set([])
        for n in self.gm.graph.nodes:
            # Create a new node and append it to the list
            target_name = self.target_names.get(n, '')
            node = MavNode(n.name, 0, 0)            
            node.operation = self.get_operation(n.op, target_name)
            node.activations = self.interp.shapes.get(n, (0,))
            node.params = self.param_counts.get(n, 0)
            node.flops = self.interp.macs.get(n, 0) * 2
            node.metadata['kwargs'] = n.kwargs         
            if n == self.err_node: node.error = True
            nodes.append(node)
            nodes_by_name[n.name] = node

            # Find connections to and from this node
            in_nodes = n.all_input_nodes
            in_node_names = [n2.name for n2 in in_nodes]
            for in_node_name in in_node_names:
                if in_node_name not in nodes_by_name: continue
                from_node = nodes_by_name[in_node_name]
                to_node = node
                if (from_node, to_node) in existing_connections: continue
                c = MavConnection(from_node, node)
                connections.append(c)
                existing_connections.add((from_node, to_node))
                    
            out_nodes = list(n.users.keys())
            out_node_names = [n2.name for n2 in out_nodes]
            for out_node_name in out_node_names:
                if out_node_name not in nodes_by_name: continue
                from_node = node
                to_node = nodes_by_name[out_node_name]
                if (from_node, to_node) in existing_connections: continue
                c = MavConnection(from_node, node)
                connections.append(c)
                existing_connections.add((from_node, to_node)) 
        
        # Assemble into graph
        self.g = MavGraph(nodes, connections)

    def summary(self) -> str:
        node_summaries : List[List[Any]] = []
        for n in self.g.nodes:
            node_summaries.append([n.name, n.operation, n.activations, n.params, n.flops])
        headers : List[str] = ['name', 'operation', 'activations', 'params', 'flops']
        return tabulate(node_summaries, headers=headers)
    

def rgetattr(m: nn.Module, attr: str) -> Tensor | None:
    # From torchinfo, used in `get_param_count()`:
    for attr_i in attr.split("."):
        if not hasattr(m, attr_i):
            return None
        m = getattr(m, attr_i)
    assert isinstance(m, Tensor)  # type: ignore[unreachable]
    return m  # type: ignore[unreachable]

def get_num_trainable_params(m:nn.Module):
    num_params = 0
    for name, param in m.named_parameters():
        # We're only looking for trainable parameters here
        if not param.requires_grad: continue

        num_params_loop = param.nelement()

        # From torchinfo `get_param_count()`:
        # Masked models save parameters with the suffix "_orig" added.
        # They have a buffer ending with "_mask" which has only 0s and 1s.
        # If a mask exists, the sum of 1s in mask is number of params.
        if name.endswith("_orig"):
            without_suffix = name[:-5]
            pruned_weights = rgetattr(m, f"{without_suffix}_mask")
            if pruned_weights is not None:
                num_params_loop = int(torch.sum(pruned_weights))
        
        num_params += num_params_loop
    return num_params
