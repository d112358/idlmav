"""
This script provides an entry point to step through the Interpreter 
functionality of `torch.fx`.
"""

# Notes:
# * In VSCode, remember to add `"justMyCode": false` to "launch.json"
# * In VSCode, hit Alt+Z on terminal to toggle line wrapping

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import warnings
import torch
from torch import nn, fx, profiler, Tensor
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from tabulate import tabulate
import torchinfo
import inspect

# ------------------------------------------------------------------------------
# Models to analyze
# ------------------------------------------------------------------------------
class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

device = 'cpu'
model = SimpleCnn().to(device)
broken_model = SimpleCnn().to(device)
broken_model.fc1 = nn.Linear(9216, 120)

input_size = (16,1,28,28)
x_input = torch.randn(input_size).to(device)

# ------------------------------------------------------------------------------
# Custom interpreter definition
# ------------------------------------------------------------------------------
class ShapeInterpreter(fx.Interpreter):
    def __init__(self, mod : torch.nn.Module):
        gm = fx.symbolic_trace(mod)
        super().__init__(gm)

        self.cur_macs: int = None
        self.shapes : Dict[fx.Node, Tuple[int]] = {}
        self.macs : Dict[fx.Node, int] = {}
        self.param_counts : Dict[fx.Node, int] = {}
        self.type_names : Dict[fx.Node,str] = {}

    def rgetattr(self, m: nn.Module, attr: str) -> Tensor | None:
        # From torchinfo, used in `get_param_count()`:
        for attr_i in attr.split("."):
            if not hasattr(m, attr_i):
                return None
            m = getattr(m, attr_i)
        assert isinstance(m, Tensor)  # type: ignore[unreachable]
        return m  # type: ignore[unreachable]

    def get_num_trainable_params(self, m:nn.Module):
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
                pruned_weights = self.rgetattr(m, f"{without_suffix}_mask")
                if pruned_weights is not None:
                    num_params_loop = int(torch.sum(pruned_weights))
            
            num_params += num_params_loop
        return num_params

    def run_node(self, n:fx.Node) -> Any:
        # Run the node
        self.cur_macs = None
        result = super().run_node(n)

        # Retrieve the shape
        if isinstance(result, Tensor):
            shape = tuple(result.shape)
        else:
            shape = (0,0,0,0)
        self.shapes[n] = shape

        # Retrieve the module type and parameter count
        if n.op == 'call_module':
            submod = self.fetch_attr(n.target)
            self.type_names[n] = submod.__class__.__name__
            self.param_counts[n] = self.get_num_trainable_params(submod)
            if self.cur_macs is not None: self.macs[n] = self.cur_macs
        if n.op == 'call_function':
            self.type_names[n] = n.target.__name__

        # Return the result
        return result
    
    def call_module(self, target, args, kwargs):
        # Run the module
        result = super().call_module(target, args, kwargs)

        # Estimate the FLOPS
        try:
            submod = self.fetch_attr(target)
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True
            ) as prof:
                submod(*args)
            flops = prof.key_averages().total_average().flops
            macs = int(flops/2)
        except Exception as e:
            warnings.warn(f'FLOPS calculation failed for module {submod.__class__.__name__}: {e}')
            macs = 0  
            # TODO: implement a fallback calculation here for well-known modules
            # e.g. https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc
        self.cur_macs = macs

        # Return the result
        return result
        
    def summary(self) -> str:
        node_summaries : List[List[Any]] = []

        for node, shape in self.shapes.items():
            type_name = self.type_names.get(node, '')
            num_params = self.param_counts.get(node, '')
            macs = self.macs.get(node, '')
            node_summaries.append(
                [node.op, node.name, type_name, node.all_input_nodes, list(node.users.keys()), shape, num_params, macs, node.target, node.args, node.kwargs])

        headers : List[str] = ['opcode', 'name', 'type', 'inputs', 'outputs', 'activations', '# params', 'MACs', 'target', 'args', 'kwargs']
        return tabulate(node_summaries, headers=headers)

# ------------------------------------------------------------------------------
# Entry point for stepping
# ------------------------------------------------------------------------------
interp = ShapeInterpreter(model)
interp.run(x_input)
print(interp.summary())

# Comparison with torchinfo
torchinfo.summary(model, input_size=x_input.shape, verbose=1, depth=8, device=device,
                  col_names=["output_size","num_params","mult_adds","input_size","kernel_size"])