from typing import Tuple, List, Dict, Set, Union, overload
from torch import nn, Tensor
import plotly.graph_objects as go
import ipywidgets as widgets
from idlmav.tracing import MavTracer
from idlmav.merging import merge_graph_nodes
from idlmav.coloring import color_graph_nodes
from idlmav.layout import layout_graph_nodes
from idlmav.release_viewer import ReleaseViewer
from idlmav.interactive_viewer import InteractiveViewer
from idlmav.mavutils import available_renderers, plotly_renderer_context

class MAV:
    def __init__(self, model:nn.Module, inputs:Union[Tensor, Tuple[Tensor]], device=None,
                 merge_threshold=0.01,
                 palette:Union[str, List[str]]='large', 
                 avoid_palette_idxs:Set[int]=set([]), 
                 fixed_color_map:Dict[str,int]={},
                 *args, **kwargs):
        self.tracer = MavTracer(model, inputs, device)
        merge_graph_nodes(self.tracer.g, 
                          cumul_param_threshold=merge_threshold)
        color_graph_nodes(self.tracer.g, 
                          palette=palette,
                          avoid_palette_idxs=avoid_palette_idxs, 
                          fixed_color_map=fixed_color_map)
        layout_graph_nodes(self.tracer.g)

    @overload
    def draw_interactive_graph(self,
                               add_table:bool=True, 
                               add_slider:bool=True, 
                               add_overview:bool=False, 
                               num_levels_displayed:float=10, 
                               height_px=400
                               ) -> widgets.Box: ...
    def draw_interactive_graph(self, *args, **kwargs) -> widgets.Box:
        return InteractiveViewer(self.tracer.g).draw(*args, **kwargs)
    
    @overload
    def draw_release_graph(self, 
                           add_table:bool=True, 
                           add_slider:bool=False, 
                           num_levels_displayed:float=10
                           ) -> go.Figure: ...
    def draw_release_graph(self, *args, **kwargs) -> go.Figure:
        return ReleaseViewer(self.tracer.g).draw(*args, **kwargs)
    
__all__ = ["MAV", "MavTracer", "merge_graph_nodes", "color_graph_nodes", "layout_graph_nodes", "ReleaseViewer", "InteractiveViewer", "available_renderers", "plotly_renderer_context"]