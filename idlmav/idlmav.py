import warnings
from typing import Tuple, Union
from torch import nn, Tensor
import plotly.graph_objects as go
import ipywidgets as widgets
from .mavoptions import MavOptions, RenderOptions
from .tracing import MavTracer
from .merging import merge_graph_nodes
from .coloring import color_graph_nodes
from .layout import layout_graph_nodes
from .renderers.figure_renderer import FigureRenderer
from .renderers.widget_renderer import WidgetRenderer
from IPython.display import display

class MAV:
    """
    High-level interface to all IDLMAV steps
    * Tracing, merging and layout are performed upon instantiation (see `MavOptions` 
      for user options)
    * Coloring and rendering are performed by calling one of several public methods 
      (see `RenderOptions` for user options)
    * This allows the user to experiment with different coloring and rendering
      options without repeating the more computationally demanding steps
    """
    def __init__(self, model:nn.Module, inputs:Union[Tensor, Tuple[Tensor]], opts:MavOptions=MavOptions(), **kwargs):
        """
        Constructs a MAV object and performs the tracing, merging and layout steps

        Parameters
        ----------
        model: nn.Module:
            PyTorch model to visualize. Must either be traceable using `torch.fx`
            or compilable using `torch.compile`

        inputs: Tensor or container or tensors
            The inputs to pass to the model's forward pass

        options: MavOptions object and/or keyword arguments
            Using a MavOptions object provides better intellisense, but 
            plain keyword arguments results in more concise code

            The following two lines are equivalent:
            ```
            mav = MAV(model, inputs, MavOptions(device='cpu', try_fx_first=False))  
            mav = MAV(model, inputs, device='cpu', try_fx_first=False)  
            ```

            See the MavOptions docstring for all available options

        Returns
        -------
        MAV object on which rendering functions are called
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        self.tracer = MavTracer(model, inputs, opts)
        merge_graph_nodes(self.tracer.g, opts)
        layout_graph_nodes(self.tracer.g, opts)

    def render_widget(self, opts:RenderOptions=RenderOptions(), **kwargs) -> widgets.Box: 
        """
        Creates an `ipywidgets.Box` object visualizing the model

        Widgets are more interactive, whereas figures are more portable.

        Options can be passed as a `RenderOptions` object and/or keyword arguments.
        Using a `RenderOptions` object provides better intellisense, but plain 
        keyword arguments results in more concise code. The following two lines 
        are equivalent:
        ```
        widget = mav.render_widget(RenderOptions(add_overview=True, palette='Vivid'))  
        widget = mav.render_widget(add_overview=True, palette='Vivid')  
        ```

        See the `RenderOptions` docstring for all available options
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        color_graph_nodes(self.tracer.g, opts)
        return WidgetRenderer(self.tracer.g).render(opts)
    
    def render_figure(self, opts:RenderOptions=RenderOptions(), **kwargs) -> go.Figure:
        """
        Creates a `plotly.graph_objects.figure` object visualizing the model

        Figures are more portable, whereas widgets are more interactive.

        Options can be passed as a `RenderOptions` object and/or keyword arguments.
        Using a `RenderOptions` object provides better intellisense, but plain 
        keyword arguments results in more concise code. The following two lines 
        are equivalent:
        ```
        fig = mav.render_figure(RenderOptions(add_overview=True, palette='Vivid'))  
        fig = mav.render_figure(add_overview=True, palette='Vivid')  
        ```

        See the `RenderOptions` docstring for all available options
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        color_graph_nodes(self.tracer.g, opts)
        return FigureRenderer(self.tracer.g).render(opts)

    def show_widget(self, opts:RenderOptions=RenderOptions(), **kwargs):
        """
        Creates and displays a widget visualizing the model

        Widgets are more interactive, whereas figures are more portable.

        Options can be passed as a `RenderOptions` object and/or keyword arguments.
        Using a `RenderOptions` object provides better intellisense, but plain 
        keyword arguments results in more concise code. The following two lines 
        are equivalent:
        ```
        mav.show_widget(RenderOptions(add_overview=True, palette='Vivid'))  
        mav.show_widget(add_overview=True, palette='Vivid')  
        ```

        See the `RenderOptions` docstring for all available options
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        widget = self.render_widget(opts)
        display(widget)
    
    def show_figure(self, opts:RenderOptions=RenderOptions(), **kwargs):
        """
        Creates and displays a figure visualizing the model

        Figures are more portable, whereas widgets are more interactive.

        Options can be passed as a `RenderOptions` object and/or keyword arguments.
        Using a `RenderOptions` object provides better intellisense, but plain 
        keyword arguments results in more concise code. The following two lines 
        are equivalent:
        ```
        mav.show_figure(RenderOptions(add_overview=True, palette='Vivid'))  
        mav.show_figure(add_overview=True, palette='Vivid')  
        ```

        See the `RenderOptions` docstring for all available options
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        fig = self.render_figure(opts)
        fig.show()

    def export_static_html(self, filename:str, opts:RenderOptions=RenderOptions(), **kwargs):
        """
        Creates and exports a figure visualizing the model

        The exported HTML file contains limited interactivity, but all interactions
        are provided on the front-end by the Plotly library. All interactions can
        therefore be enjoyed without a running kernel or backend.

        Options can be passed as a `RenderOptions` object and/or keyword arguments.
        Using a `RenderOptions` object provides better intellisense, but plain 
        keyword arguments results in more concise code. The following two lines 
        are equivalent:
        ```
        mav.export_static_html(RenderOptions(add_overview=True, export_for_offline_use=True))  
        mav.export_static_html(add_overview=True, export_for_offline_use=True)  
        ```

        See the `RenderOptions` docstring for all available options
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        fig = self.render_figure(opts)
        include_plotlyjs = True if opts.export_for_offline_use else 'cdn'
        fig.write_html(filename, include_plotlyjs=include_plotlyjs)


    def export_html(self, filename:str, opts:RenderOptions=RenderOptions(), **kwargs):
        """Deprecated method. Use `export_static_html` instead."""
        warnings.warn(
            "`export_html` is deprecated and will be removed in a future release. Use `export_static_html` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        self.export_static_html(filename, opts)