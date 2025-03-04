from .mavoptions import MavOptions
from .mavtypes import MavNode, MavConnection, MavGraph
from .mavutils import to_device
from typing import Any, Dict, List, Tuple, Set, Optional, Mapping
from collections import deque
import warnings
import torch
from torch import nn, fx, profiler, Tensor
import torch.nn.functional as F
import re
from tabulate import tabulate

class ShapeMacInterpreter(fx.Interpreter):
    """
    Class that performs `torch.fx` interpretation to extract
    activations and FLOPS for every node in the graph
    """
    def __init__(self, gm : fx.GraphModule):
        super().__init__(gm)

        # Outputs
        self.shapes : Dict[fx.Node, Tuple[int]] = {}
        self.flops : Dict[fx.Node, int] = {}

        # State
        self.running_node = None
        self.last_successful_node = None
        self.cur_flops: int = None

    def run_node(self, n:fx.Node) -> Any:
        # Run the node
        self.cur_flops = None
        self.running_node = n
        result = super().run_node(n)
        self.running_node = None

        # Retrieve the shape
        if isinstance(result, Tensor):
            shape = tuple(result.shape)
        else:
            shape = (0,0,0,0)
        self.shapes[n] = shape

        # Store the number of FLOPS if calculated
        if n.op == 'call_module' or n.op == 'call_function':
            if self.cur_flops is not None: self.flops[n] = self.cur_flops

        # Update the state and return the result
        self.last_successful_node = n
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
        except Exception as e:
            warnings.warn(f'FLOPS calculation failed for module {submod.__class__.__name__}: {e}')
            flops = 0  
        self.cur_flops = flops

        # Return the result
        return result
        
    def call_function(self, target, args, kwargs):
        # Run the module
        result = super().call_function(target, args, kwargs)

        # Estimate the FLOPS
        try:
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True
            ) as prof:
                target(*args, **kwargs)
            flops = prof.key_averages().total_average().flops
        except Exception as e:
            warnings.warn(f'FLOPS calculation failed for function {target.__name__}: {e}')
            flops = 0  
        self.cur_flops = flops

        # Return the result
        return result

class MavTracer:
    """
    Class that performs the tracing step

    All processing is performed upon instantiation

    After processing, the `MavGraph` object required for the remaining steps
    may be found in the `g` class member.
    """
    def __init__(self, model:nn.Module, inputs:Any, 
                 opts:MavOptions=MavOptions(), **kwargs):
        """
        Creates an instance of `MavTracer` and performs the tracing step

        Keyword arguments may be passed either via a `MavOptions` object or
        as-is. Using a `MavOptions` object provides better intellisense, 
        but plain keyword arguments results in more concise code.

        The following two lines are equivalent:
        ```
        tracer = MavTracer(model, inputs, MavOptions(device='cpu', try_fx_first=False))  
        tracer = MavTracer(model, inputs, device='cpu', try_fx_first=False)  
        ```

        Parameters
        ----------
        model: nn.Module:
            PyTorch model to trace. Must either be traceable using `torch.fx`
            or compilable using `torch.compile`

        inputs: Tensor or container or tensors
            The inputs to pass to the model's forward pass

        device: str or None:
            If not None, moves the model and inputs to the specified device, 
            e.g. 'cpu', 'cuda'
            
        try_fx_first: bool
            Specifies whether to first attempt tracing the computation graph using 
            `torch.fx.symbolic_trace` before falling back to `torch.compile`. 
            * `torch.fx.symbolic_trace` fails more often than `torch.compile`, but 
              when it passes, classes in the model are preserved.
            * For example, `torch.fx.symbolic_trace` will produce an `nn.Conv2d`
              module where `torch.compile` will produce a `conv2d()` function call
              with trainable parameters in an external node.

        keep_internal_nodes: bool
            After tracing with `torch.compile`, some nodes represent trainable 
            parameters, buffers, constants and manipulations of these that are 
            usually considered internal to a module. Let's call these internal
            nodes and define them as nodes outside the main branch (the set of all 
            nodes reachable from nodes representing inputs to the model)
            * If set to `False`, IDLMAV attempts to propagate internal nodes to the 
            first operation on the main branch that uses them.
            * If set to `True`, internal nodes are reported as-is in the final graph.  
            
            `keep_internal_nodes` is applicable to computational graphs traced with
            `torch.compile` (either as a result of `torch.fx.symbolic_trace failing
            or of setting `try_fx_first` to False).

        concrete_args: dict[str, any]
            If specified, this argument is passed as-is to `torch.fx.symbolic_trace` 
            to fix some of the arguments to the forward pass method. See
            the documentation of `torch.fx.symbolic_trace` for more information
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        if opts.device:
            self.model = model.to(opts.device)
            self.inputs = to_device(inputs, opts.device)
        else:
            self.model = model
            self.inputs = inputs
        self.concrete_args = opts.concrete_args
        self.try_fx_first = opts.try_fx_first
        self.keep_internal_nodes = opts.keep_internal_nodes
        self.graphs:List[Tuple[fx.GraphModule, List[torch.Tensor]]] = []
        self.gm: fx.GraphModule = None
        self.interp: ShapeMacInterpreter = None
        self.g: MavGraph = None
        self.input_sizes:Mapping[fx.Node,int] = {}
        self.param_counts : Mapping[fx.Node,int] = {}
        self.target_names : Mapping[fx.Node,str] = {}
        self.entry_types : Mapping[fx.Node,str] = {}
        self.exit_types : Mapping[fx.Node,str] = {}
        self.propagated_params : Mapping[fx.Node,int] = {}
        self.err_node: fx.Node = None
        self.long_name_replacements:Mapping[str,str] = {}
        self.long_name_repl_pattern:re.Pattern = None
        self.run()
        self.build_graph()

    def run(self):
        if self.try_fx_first:
            try:
                self.run_fx()
            except Exception as e:
                print(f'Tracing failed with torch.fx.symbolic_trace: {e}')
                print('Tracing with torch.compile')
                self.run_compiler()
        else:
            self.run_compiler()

    def run_fx(self):
        # 1st pass: symbolic tracing using torch.fx
        self.gm = fx.symbolic_trace(self.model, self.concrete_args)
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
            if isinstance(self.inputs, Mapping):
                self.interp.run(**self.inputs)
            elif isinstance(self.inputs, Tuple):
                self.interp.run(*self.inputs)
            else:
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

    def custom_compiler_backend(self, gm:fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.graphs.append((gm, example_inputs))
        # According to `fx.Interpreter.run()`, positional function args are consumed left-to-right by `placeholder` nodes.
        x_iter = iter(example_inputs)
        for n in gm.graph.nodes:
            if n.op != 'placeholder': continue
            x = next(x_iter)
            self.input_sizes[n] = x.nelement()
        return gm.forward
    
    def run_compiler(self):
        # 1st pass: Compile to intercept all fx graphs
        torch._dynamo.reset()
        compiled_model = torch.compile(self.model, backend=self.custom_compiler_backend)
        if isinstance(self.inputs, Mapping):
            outputs = compiled_model(**self.inputs)
        elif isinstance(self.inputs, Tuple):
            outputs = compiled_model(*self.inputs)
        else:
            outputs = compiled_model(self.inputs)
        
        # For now, select the largest graph based on number of `call_function` operations
        # * TODO: If proved valuable for many popular models, implement user controls to manually select graph
        gm_lengths = [len([n for n in gm.graph.nodes if n.op == 'call_function']) for gm,xs in self.graphs]
        gm_idx = gm_lengths.index(max(gm_lengths))
        self.gm, xs = self.graphs[gm_idx]
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
                elif n.op == 'call_method':
                    self.target_names[n] = n.target
        except Exception as e:
            self.err_node = n
            warnings.warn(f'2nd tracing pass failed for module {n.target}: {e}')

        # 3rd pass: forward pass using torch.fx.Interpreter
        try:
            self.interp.run(*xs)
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

    def calc_entry_exit_types(self) -> Tuple[Mapping[fx.Node, str], Mapping[fx.Node, str], Mapping[fx.Node, int]]:
        """
        Defines an entry and exit type for each node in the graph
        * `entry_type=='normal'`: Node is downstream from a model input node 
        * `entry_type=='learnable'`: Node is downstream from a node representing learnable parameters 
        * `entry_type=='learn-split`': Node is downstream from a sub-branch of a learnable parameter node, or split occurs at node
        * `entry_type=='buffer'`: Node is downstream from a node representing a buffer
        * `entry_type=='special': Node is not downstream from an input node or learnable parameter node 
        * `exit_type=='normal`': Node is upstream from a model output node
        * `exit_type=='special'`: Node is not upstream from a model output node

        Updates 3 dicts:
        * `self.entry_types[n]` is a string specifying the entry type of node `n`, as defined above
        * `self.exit_types[n]` is a string specifying the exit type of node `n`, as defined above
        * `self.propagated[n]` is an int specifying the number of parameters of node `n`, as propagated from learnable
           parameter nodes up to either a split or a normal entry node, whichever is encountered first
        """
        learnable_placeholder_nodes = [n for n in self.gm.graph.nodes if n.op == 'placeholder' and n.name.startswith('l_self_') and '_parameters_' in n.name]
        buffer_placeholder_nodes = [n for n in self.gm.graph.nodes if n.op == 'placeholder' and n.name.startswith('l_self_') and '_parameters_' not in n.name]
        input_placeholder_nodes = [n for n in self.gm.graph.nodes if n.op == 'placeholder' and not n.name.startswith('l_self_')]
        output_nodes = [n for n in self.gm.graph.nodes if n.op == 'output']
        self.entry_types = {}  # ['normal','learnable','learn-split','special']
        self.exit_types = {}  # ['normal','special']
        self.propagated_params = {}  # Assign to first 'normal' or 'split' node
        
        # BFS on input placeholder nodes
        for n in input_placeholder_nodes: self.entry_types[n] = 'normal'
        queue = deque(input_placeholder_nodes)  # Initialize to contain all output nodes
        while queue:
            n = queue.popleft()
            out_nodes = list(n.users.keys())
            for out_node in out_nodes:
                if out_node in self.entry_types: continue  # Already traversed
                self.entry_types[out_node] = 'normal'
                queue.append(out_node)

        # BFS on learnable placeholder nodes
        for n in learnable_placeholder_nodes: self.entry_types[n] = 'learnable'
        queue = deque(learnable_placeholder_nodes)  # Initialize to contain all output nodes
        temp_sizes = {k:v for k,v in self.input_sizes.items()}
        while queue:
            n = queue.popleft()
            input_size = temp_sizes.get(n,0)
            out_nodes = list(n.users.keys())

            # Detect first split
            if len(out_nodes) > 1 and self.entry_types[n] == 'learnable':
                self.entry_types[n] = 'learn-split'
                self.propagated_params[n] = self.propagated_params.get(n,0) + input_size

            # Propagate to output nodes and queue them
            for out_node in out_nodes:
                if out_node in self.entry_types:
                    # Already traversed, so stop processing, but assign propagated input size if not assigned upstream yet
                    if self.entry_types[n] == 'learnable': 
                        if self.entry_types[out_node] == 'learnable':
                            temp_sizes[out_node] = temp_sizes.get(out_node,0) + input_size
                        else:
                            self.propagated_params[out_node] = self.propagated_params.get(out_node,0) + input_size
                    continue
                self.entry_types[out_node] = self.entry_types[n]
                if self.entry_types[n] == 'learnable': temp_sizes[out_node] = temp_sizes.get(out_node,0) + input_size
                queue.append(out_node)
        
        # BFS on buffer placeholder nodes
        for n in buffer_placeholder_nodes: self.entry_types[n] = 'buffer'
        queue = deque(buffer_placeholder_nodes)  # Initialize to contain all output nodes
        while queue:
            n = queue.popleft()
            out_nodes = list(n.users.keys())
            for out_node in out_nodes:
                if out_node in self.entry_types: continue  # Already traversed
                self.entry_types[out_node] = self.entry_types[n]
                queue.append(out_node)
        
        # BFS on output nodes
        for n in output_nodes: self.exit_types[n] = 'normal'
        queue = deque(output_nodes)  # Initialize to contain all output nodes
        while queue:
            n = queue.popleft()
            in_nodes = n.all_input_nodes
            for in_node in in_nodes:
                if in_node in self.exit_types: continue  # Already traversed
                self.exit_types[in_node] = 'normal'
                queue.append(in_node)

        # Special entry and exit nodes
        for n in self.gm.graph.nodes:
            if n not in self.entry_types: self.entry_types[n] = 'special'
            if n not in self.exit_types: self.exit_types[n] = 'special'

    def get_operation(self, n:fx.Node):
        if self.entry_types.get(n,'unknown') == 'learn-split':
            input_entry_types = [self.entry_types.get(in_node, 'unknown') for in_node in n.all_input_nodes]
            if not any([t=='learn-split' for t in input_entry_types]): return 'Shared params'
        target_name = self.target_names.get(n, '')
        match n.op:
            case 'placeholder': return 'input'
            case 'output': return 'output'
            case 'call_module': return f'nn.{target_name}'
            case 'call_function': return f'{target_name}()'
            case 'call_method': return f'.{target_name}()'
            case _: return target_name
        
    # def parse_args(self, x:Any):
    #     if isinstance(x, Mapping): return {k:self.parse_args(v) for k,v in x.items()}
    #     if isinstance(x, list): return [self.parse_args(o) for o in x]
    #     if isinstance(x, tuple): return tuple(self.parse_args(list(x)))
    #     if isinstance(x, fx.Node): return x
    #     return x

    def ensure_unique(self, name, existing_names):
        match = re.match(r"^(.*?)(\d+)?$", name)
        base, num = match.groups()
        num = int(num) if num else 0
        new_name = name

        while new_name in existing_names:
            num += 1
            new_name = f"{base}{num}"
        return new_name
    
    def build_long_name_replacements(self):
        long_names = [n.name for n in self.gm.graph.nodes if n.op == 'placeholder' and n.name.startswith('l_self_')]
        self.long_name_replacements:Mapping[str,str] = {}
        for long_name in long_names:
            short_name = re.sub('_self|_modules|_parameters','',long_name)
            short_name = self.ensure_unique(short_name, self.long_name_replacements.values())
            self.long_name_replacements[long_name] = short_name
        self.long_name_repl_pattern = re.compile("|".join(map(re.escape, self.long_name_replacements.keys())))

    def long_name_replace_match(self, match):        
        return self.long_name_replacements[match.group(0)]
        
    def shorten(self, text):
        if not self.long_name_replacements: return text
        return self.long_name_repl_pattern.sub(self.long_name_replace_match, text)

    def build_graph(self):
        self.build_long_name_replacements()
        self.calc_entry_exit_types()
        if self.keep_internal_nodes: 
            prop_params = {k:v for k,v in self.input_sizes.items() if self.entry_types.get(k,'')=='learnable' or self.entry_types.get(k,'')=='learn-split'}
        else:
            prop_params = self.propagated_params
        nodes: List[MavNode] = []
        nodes_by_name: Mapping[str, MavNode] = {}
        connections: List[MavConnection] = []
        existing_connections: Set[Tuple[MavNode, MavNode]] = set([])
        for n in self.gm.graph.nodes:
            # Create a new node and append it to the list
            entry_type = self.entry_types.get(n, 'unknown')
            exit_type = self.exit_types.get(n, 'unknown')
            if entry_type != 'normal' and entry_type != 'learn-split' and not self.keep_internal_nodes: continue
            if exit_type != 'normal': continue
            short_name = self.shorten(n.name)
            node = MavNode(short_name, 0, 0)            
            node.operation = self.get_operation(n)
            if not node.operation:
                break_here = True
            node.activations = self.interp.shapes.get(n, (0,))
            node.params = self.param_counts.get(n, 0) + prop_params.get(n, 0)
            node.flops = self.interp.flops.get(n, 0)
            node.metadata['args'] = self.shorten(n.args.__repr__())
            node.metadata['kwargs'] = n.kwargs.__repr__()
            node.metadata['fx_name'] = n.name
            node.metadata['entry_type'] = entry_type
            node.metadata['exit_type'] = exit_type
            if n == self.err_node: node.error = True
            nodes.append(node)
            nodes_by_name[short_name] = node

            # Find connections to and from this node
            in_nodes = n.all_input_nodes
            in_node_names = [self.shorten(n2.name) for n2 in in_nodes]
            for in_node_name in in_node_names:
                if in_node_name not in nodes_by_name: continue
                from_node = nodes_by_name[in_node_name]
                to_node = node
                if (from_node, to_node) in existing_connections: continue
                c = MavConnection(from_node, node)
                connections.append(c)
                existing_connections.add((from_node, to_node))
                    
            out_nodes = list(n.users.keys())
            out_node_names = [self.shorten(n2.name) for n2 in out_nodes]
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
    # From torchinfo, used in `get_num_trainable_params()`:
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
