from typing import Tuple, List, Dict, Set, Union, Optional, Any, overload
from torch import nn, Tensor
import plotly.graph_objects as go
import ipywidgets as widgets
from .tracing import MavTracer
from .merging import merge_graph_nodes
from .coloring import color_graph_nodes
from .layout import layout_graph_nodes
from .renderers.figure_renderer import FigureRenderer
from .renderers.widget_renderer import WidgetRenderer
from IPython.display import display

class MAV:
    def __init__(self, model:nn.Module, inputs:Union[Tensor, Tuple[Tensor]], device=None,
                 merge_threshold=0.01,
                 concrete_args: Optional[Dict[str, Any]] = None,
                 show_param_nodes=False):
        self.tracer = MavTracer(model, inputs, device, concrete_args, show_param_nodes)
        merge_graph_nodes(self.tracer.g, 
                          cumul_param_threshold=merge_threshold)
        layout_graph_nodes(self.tracer.g)

    def render_widget(self,
                      add_table:bool=True, 
                      add_slider:bool=True, 
                      add_overview:bool=False, 
                      num_levels_displayed:float=10, 
                      height_px=400,
                      palette:Union[str, List[str]]='large', 
                      avoid_palette_idxs:Set[int]=set([]), 
                      fixed_color_map:Dict[str,int]={},
                      ) -> widgets.Box: 
        color_graph_nodes(self.tracer.g, palette=palette,avoid_palette_idxs=avoid_palette_idxs, fixed_color_map=fixed_color_map)
        return WidgetRenderer(self.tracer.g).render(add_table=add_table, add_slider=add_slider, add_overview=add_overview, 
                                                    num_levels_displayed=num_levels_displayed, height_px=height_px)
    
    def render_figure(self, 
                      add_table:bool=True, 
                      add_slider:bool=False, 
                      add_overview:bool=False, 
                      num_levels_displayed:float=10,
                      palette:Union[str, List[str]]='large', 
                      avoid_palette_idxs:Set[int]=set([]), 
                      fixed_color_map:Dict[str,int]={},
                      ) -> go.Figure:
        color_graph_nodes(self.tracer.g, palette=palette,avoid_palette_idxs=avoid_palette_idxs, fixed_color_map=fixed_color_map)
        return FigureRenderer(self.tracer.g).render(add_table=add_table, add_slider=add_slider, add_overview=add_overview,
                                                    num_levels_displayed=num_levels_displayed)

    @overload
    def show_widget(self,
                    add_table:bool=True, 
                    add_slider:bool=True, 
                    add_overview:bool=False, 
                    num_levels_displayed:float=10, 
                    height_px=400,
                    palette:Union[str, List[str]]='large', 
                    avoid_palette_idxs:Set[int]=set([]), 
                    fixed_color_map:Dict[str,int]={},
                    ): ...
    def show_widget(self, *args, **kwargs):
        widget = self.render_widget(*args, **kwargs)
        display(widget)
    
    @overload
    def show_figure(self, 
                    add_table:bool=True, 
                    add_slider:bool=False, 
                    add_overview:bool=False, 
                    num_levels_displayed:float=10,
                    palette:Union[str, List[str]]='large', 
                    avoid_palette_idxs:Set[int]=set([]), 
                    fixed_color_map:Dict[str,int]={},
                    ): ...
    def show_figure(self, *args, **kwargs):        
        fig = self.render_figure(*args, **kwargs)
        fig.show()

    @overload
    def export_html(self, 
                    filename,
                    offline=False,
                    add_table:bool=True, 
                    add_slider:bool=False, 
                    add_overview:bool=False, 
                    num_levels_displayed:float=10,
                    palette:Union[str, List[str]]='large', 
                    avoid_palette_idxs:Set[int]=set([]), 
                    fixed_color_map:Dict[str,int]={},
                    ): ...
    def export_html(self, filename, offline=False, *args, **kwargs):        
        fig = self.render_figure(*args, **kwargs)
        include_plotlyjs = True if offline else 'cdn'
        fig.write_html(filename, include_plotlyjs=include_plotlyjs)
