from typing import List, Mapping, Set, Any, Union
from dataclasses import dataclass, field

@dataclass
class MavOptions:
    """
    Parameters
    ----------
    device: str or None
        If not None, moves the model and inputs to the specified device, 
        e.g. 'cpu', 'cuda'

    merge_threshold: float
        Determines the amount of merging to perform:
        * Negative values disable merging altogether
        * A value of zero causes only nodes without any parameters to be merged
        * For values between 0 and 1, all nodes will be sorted in ascending order 
          of the number of parameters and merged from the smallest node until
          a cumulative fraction of merge_threshold is reached
        * The default value of 0.01 typically causes nodes without parameters
          and very small nodes such as normalization operations to be merged

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
    device: str = None
    merge_threshold: float = 0.01
    try_fx_first: bool = True
    keep_internal_nodes: bool = False
    concrete_args: Mapping[str, Any] = None

@dataclass
class RenderOptions:
    """
    Parameters
    ----------
    add_table: bool
        Specifies whether to include the table on the right
        * The table summarizes the total number of learnable parameters and FLOPS
          used by the model, as well as the totals for each operation and the
          activations after each operation

    add_slider: bool
        Specifies whether to include a slider
        * Without the` slider, panning and zooming are still possible using the
          build-in controls provided by plotly
        * For `show_widget`, the slider is displayed to the left of the figure
          and is synchronized with plotly's built-in pan and zoom controls
        * For `show_figure`, a horizontal slider is displayed below the figure
          and is not synchronized with plotly's built-in pan and zoom controls.
          Both these are limitations of the frontend-only slider provided by
          plotly.
    
    add_overview: bool
        Specifies whether to include an overview panel to indicate where the 
        main panel is currently zoomed to
        * The overview panel is always displayed to the left of the main panel
        * For `show_widget`, the overview panel is synchronized with the slider
          and plotly's built-in pan and zoom controls
        * For `show_figure`, the overview panel only responds to slider actions

    num_levels_displayed: int
        The initial number of levels to display in the zoomed-in main panel
        * Static figures ignore this option and displays the whole graph id
          `add_slider` is False
        * After creating the figure, pan and zoom controls may be used to 
          change the number of levels displayed

    height_px: int:
        The height of the figure in pixels

    palette: string or list of strings
        A discrete color palette to use for node marker colors when coloring
        by operation. The value may be in any of the following formats:
        * A named palette from https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express, e.g. 'Vivid'
        * A list of strings in '#RRGGBB' format
        * One of the strings 'large' or 'small'

    avoid_palette_idxs: set[int]
        Indices in the specified `palette` that should not be used for marker colors. 
        This is useful when specifying a named palette from https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express

    fixed_color_map: dict[str,int]
        Force specific node operations to specific colors to ensure consistency
        across models visualized.
        
        Keys may take on any of the following formats:
        * Any value in the "Operations" column of the table produced by IDLMAV
        * Any category listed at https://pytorch.org/docs/stable/nn.html

        Example:
        ```
        fixed_color_map={'Convolution':7, 'add()':0, 'nn.MaxPool2d':5}
        ```

    continuous_colorscale: any
        This palette is used when coloring nodes by the number of parameters or
        FLOPS. The value is passed as-is to the `colorscale` field of the
        `marker` object (see https://plotly.com/python/colorscales/#color-scale-for-scatter-plots-with-graph-objects).
        It can take any form accepted by Plotly, but the easiest is a single
        string such as "Viridis", "Thermal", etc. See https://plotly.com/python/builtin-colorscales/
        for more options.

    size_color_idx: int or None
        Determines the criteria used for the size and color of node markers:
        * 0: Size by number of parameters, color by operation 
        * 1: Size by number of FLOPS, color by operation 
        * 2: Size by number of parameters, color by number of FLOPS 
        * 3: Size by number of FLOPS, color by number of parameters

        This can also be changed interactively using a dropdown menu. This 
        parameter simply determines the initial state of the dropdown menu.

        If unassigned or None, this defaults to 1 if `keep_internal_nodes` was
        selected during tracing or 0 otherwise.

    export_for_offline_use: bool
        Specifies how to include Plotly library in exported HTML
        * If `True`, the entire plotly library (approximately 4 MB) is included
          in the exported HTML
        * If `False`, Plotly is linked into the exported HTML using a CDN
        This option is only applicable to methods that export rendered graphs to HTML
    """
    add_table: bool = True
    add_slider: bool = False
    add_overview: bool = False
    num_levels_displayed: float = 10
    height_px: int = 400
    palette: Union[str, List[str]] = 'large'
    avoid_palette_idxs: Set[int] = set([]),
    fixed_color_map: Mapping[str,int] = field(default_factory=dict)
    continuous_colorscale: Any = 'Bluered'
    size_color_idx: int = None
    export_for_offline_use: bool = False