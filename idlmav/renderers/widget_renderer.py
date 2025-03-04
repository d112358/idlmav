from ..mavoptions import RenderOptions
from ..mavtypes import MavNode, MavGraph, MavConnection
from .renderer_utils import use_straight_connection, segmented_line_coords
import time
import numpy as np
import plotly.graph_objects as go
import plotly.callbacks as cb
from plotly.basedatatypes import BaseTraceType
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript

class WidgetRenderer:
    """
    This viewer uses `go.FigureWidget` and other `ipywidgets`, 
    providing more interactions during development. Unfortunately,
    these require a running kernel to display correctly. When
    uploading a notebook to GitHub, therefore, FigureRenderer
    is recommended.

    Available interactions and limitations:
    * Hover over modules to see activation sizes, number of parameters 
      and FLOPS
    * An optional table is available with synchronized scrolling
      between the table and the graph
    * Clicking on a module highlights that module in the table
    * An optional overview window displays a zoomed out copy of the 
      model. Clicking on the overview window pans the zoomed in copy
      of the model to the clicked area and scrolls the table 
      accordingly
    * An optional vertical range slider may be added for additional
      control of synchronized scrolling
    """

    def __init__(self, g:MavGraph):
        self.g = g

        # Panels and widgets
        self.main_panel      : widgets.Box      = None
        self.table_panel     : widgets.Box      = None
        self.overview_panel  : widgets.Box      = None
        self.slider_panel    : widgets.Box      = None
        self.main_fig        : go.FigureWidget  = None
        self.table_widget    : widgets.Output   = None
        self.overview_fig    : go.FigureWidget  = None
        self.slider_widget   : widgets.FloatRangeSlider = None
        self.dropdown_widget : widgets.Dropdown = None

        # Traces
        self.overview_rect_trace   : BaseTraceType = None
        self.sel_marker_trace      : BaseTraceType = None
        self.node_marker_trace     : BaseTraceType = None
        self.overview_marker_trace : BaseTraceType = None

        # Options
        self.size_color_options = [('params','operation'),
                                   ('flops','operation'),
                                   ('params','flops'),
                                   ('flops','params')]
        self.continuous_colorscale = 'Bluered'
        self.size_color_idx: int = None

        # Derived parameters
        self.in_level = min([n.y for n in g.in_nodes])
        self.out_level = max([n.y for n in g.out_nodes])
        self.graph_num_rows = self.out_level - self.in_level + 1
        self.full_y_range = [self.out_level+0.5, self.in_level-0.5]  # Note the reversed order: plotting input at the top
        self.min_x = min([n.x for n in g.nodes])
        self.max_x = max([n.x for n in g.nodes])
        self.graph_num_cols = self.max_x - self.min_x + 1
        self.full_x_range = [self.min_x-0.5, self.max_x+0.5]
        
        all_params = [n.params for n in g.nodes]
        all_flops = [n.flops for n in g.nodes]
        pos_params = [v for v in all_params if v > 0]
        pos_flops = [v for v in all_flops if v > 0]        
        self.params_range = [min(pos_params), max(pos_params)] if pos_params else [0,0]
        self.flops_range = [min(pos_flops), max(pos_flops)] if pos_flops else [0,0]
        self.params_log_ratio = np.log2(self.params_range[1]) - np.log2(self.params_range[0]) if pos_params else None
        self.flops_log_ratio = np.log2(self.flops_range[1]) - np.log2(self.flops_range[0]) if pos_flops else None

        # State variables
        self.unique_id = f'{id(self)}_{int(time.time() * 1000)}'
        self.updating_slider = False

    def render(self, opts:RenderOptions=RenderOptions(), **kwargs) -> widgets.Box:
        """
        Renders the graph received during construction as a `ipywidgets.Box` object

        Keyword arguments may be passed either via a `RenderOptions` object or
        as-is. Using a `RenderOptions` object provides better intellisense, 
        but plain keyword arguments results in more concise code.

        The following two lines are equivalent:
        ```
        widget = WidgetRenderer(graph).render(RenderOptions(add_overview=True, height_px=500)  
        widget = WidgetRenderer(graph).render(add_overview=True, height_px=500)  
        ```

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

        Returns
        -------       
        `ipywidgets.Box` object that can be displayed using `IPython.display`
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        
        # Setup parameters
        g = self.g
        self.continuous_colorscale = opts.continuous_colorscale
        self.size_color_idx = opts.size_color_idx
        initial_y_range = self.fit_range([self.in_level+opts.num_levels_displayed-0.5, self.in_level-0.5], self.full_y_range)
        initial_x_range = self.full_x_range

        # Create a new unique ID every time this is called        
        self.unique_id = f'{id(self)}_{int(time.time() * 1000)}'
    
        # Create the main panel
        main_panel_layout = widgets.Layout(flex = '0 1 auto', margin='0px', padding='0px', overflow='hidden')
        main_fig_layout = go.Layout(
            width=max((self.graph_num_cols*100, 180)), height=opts.height_px,
            plot_bgcolor='#e5ecf6',
            autosize=True,
            xaxis=dict(range=initial_x_range, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=initial_y_range, showgrid=False, zeroline=False, visible=False),
            margin=dict(l=0, r=2, t=1, b=1),
            showlegend=False,
            title=dict(text=None)
        )        
        self.main_fig = go.FigureWidget(layout=main_fig_layout)
        self.main_panel = widgets.Box(children=[self.main_fig], layout=main_panel_layout)
        panels = [self.main_panel]
        
        # Add a selection marker (behind notes for hover purposes)
        node = g.nodes[0]
        sel_marker = go.Scatter(
            x=[node.x], y=[node.y], 
            mode='markers', 
            marker=dict(
                size=[self.params_to_dot_size(node.params)],
                color='rgba(0,0,0,0.1)',
                line=dict(color='black', width=3)
            ),
            hoverinfo='skip', showlegend=False
        )
        self.main_fig.add_trace(sel_marker)
        self.sel_marker_trace = self.main_fig.data[-1]

        # Draw connections lines between the nodes
        # * Use a single trace with `None` values separating different lines
        # * Using a separate trace for every line cause a blank display 
        #   on Colab
        # * Separate traces may also negatively impact responsiveness, e.g. 
        #   to pan & zoom actions
        x_coords, y_coords = [],[]
        for c in g.connections:
            xs, ys = self.get_connection_coords(c)
            if x_coords: x_coords.append(None)
            if y_coords: y_coords.append(None)
            x_coords += xs
            y_coords += ys

        line_trace = go.Scatter(
            x=x_coords, y=y_coords, mode="lines",
            line=dict(color="gray", width=1),
            hoverinfo='skip',
            showlegend=False
        )
        self.main_fig.add_trace(line_trace)

        # Draw nodes
        if self.size_color_idx is None:
            total_non_input_params = sum([n.params for n in g.nodes if n not in g.in_nodes])
            total_non_input_flops = sum([n.flops for n in g.nodes if n not in g.in_nodes])
            initially_size_by_flops = total_non_input_params == 0 and total_non_input_flops > 0
            self.size_color_idx = 1 if initially_size_by_flops else 0
        init_size_by, init_color_by = self.size_color_options[self.size_color_idx]
        node_trace = self.build_node_trace(False, init_size_by, init_color_by)
        self.main_fig.add_trace(node_trace)
        self.node_marker_trace = self.main_fig.data[-1]

        # Add table if selected
        if opts.add_table:
            table_panel_layout = widgets.Layout(flex='0 0 auto', margin='0px', padding='0px', overflow='visible')
            table_style = self.write_table_style()
            table_html = self.write_table_html(g)
            scrolling_table_html = f'<div id="{self.html_scrolling_table_id()}" style="height: {opts.height_px}px; overflow: auto; width: fit-content">{table_html}</div>'
            self.table_widget = widgets.Output()
            with self.table_widget:
                display(HTML(table_style))
                display(HTML(scrolling_table_html))
            self.table_panel = widgets.Box(children=[self.table_widget], layout=table_panel_layout)
            panels.append(self.table_panel)
            
        # Add overview window if selected
        if opts.add_overview:
            # Overview panel
            overview_panel_layout = widgets.Layout(flex = '0 0 auto', margin='0px', padding='0px', overflow='hidden')
            overview_fig_layout = go.Layout(
                width=max((self.graph_num_cols*15, 45)),
                height=opts.height_px,
                plot_bgcolor='#dfdfdf',
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=self.full_y_range, 
                           showgrid=False, zeroline=False, visible=False),
                margin=dict(l=0, r=4, t=1, b=1),
                showlegend=False,
                title=dict(text=None),
                hoverdistance=-1,  # Always hover over something
            )
            self.overview_fig = go.FigureWidget(layout=overview_fig_layout)
            self.overview_panel = widgets.Box(children=[self.overview_fig], layout=overview_panel_layout)
            panels.insert(0, self.overview_panel)

            # Connection lines
            self.overview_fig.add_trace(line_trace)
                
            # Nodes
            overview_nodes_trace = self.build_node_trace(True, init_size_by, init_color_by)
            self.overview_fig.add_trace(overview_nodes_trace)
            self.overview_marker_trace = self.overview_fig.data[-1]

            # Rectangle
            x0, y0, x1, y1 = self.min_x-0.5, initial_y_range[0], self.max_x+0.5, initial_y_range[1]
            rect_trace = go.Scatter(
                x=[x0, x0, x1, x1, x0], 
                y=[y0, y1, y1, y0, y0], 
                fill='toself',
                mode='lines',
                line=dict(color="#3d6399", width=1),
                fillcolor='rgba(112,133,161,0.25)',
                hoveron='points',
                hoverinfo='skip',
                showlegend=False
            )
            self.overview_fig.add_trace(rect_trace)
            self.overview_rect_trace = self.overview_fig.data[-1]

        # Add slider if selected
        # * Use negative values everywhere, because ipywidgets does not support
        #   inverting the direction of vertical sliders
        if opts.add_slider:
            slider_panel_layout = widgets.Layout(flex = '0 0 auto', margin='0px', padding='0px', overflow='visible')
            self.slider_widget = widgets.FloatRangeSlider(
                value=[-initial_y_range[0], -initial_y_range[1]], min=-self.full_y_range[0], max=-self.full_y_range[1], 
                step=0.01, description='', orientation='vertical', continuous_update=True,
                layout=widgets.Layout(height=f'{opts.height_px}px')
            )
            self.slider_widget.readout = False  # For some reason it does not seem to work if set during construction
            self.slider_panel = widgets.Box(children=[self.slider_widget], layout=slider_panel_layout)
            panels.insert(0, self.slider_panel)

        # Add dropdown menu for marker sizes and colors
        size_color_labels = dict(operation='operation', params='params', flops='FLOPS')
        dropdown_options = []
        for size_by, color_by in self.size_color_options:
            label = f'Size by {size_color_labels[size_by]}, color by {size_color_labels[color_by]}'
            dropdown_options.append((label, (size_by, color_by)))
        self.dropdown_widget = widgets.Dropdown(options=dropdown_options, value=(init_size_by,init_color_by), description='')

        # Create container for all panels
        # * To be displayed in Notebook using `display`
        horz_container_layout = widgets.Layout(width='100%', margin='0px', padding='0px')
        horz_container = widgets.HBox(panels, layout=horz_container_layout)
        container_layout = widgets.Layout(margin='0px', padding='0px')
        container = widgets.VBox([horz_container, self.dropdown_widget], layout=container_layout)

        # Set up event handlers        
        self.node_marker_trace.on_click(self.on_main_panel_click)
        self.main_fig.layout.on_change(self.on_main_panel_pan_zoom, 'xaxis.range', 'yaxis.range')
        if self.overview_fig:
            self.overview_marker_trace.on_click(self.on_overview_panel_click)
        if self.slider_widget:
            self.slider_widget.observe(self.on_slider_value_change, names="value")
        if self.dropdown_widget:
            self.dropdown_widget.observe(self.on_dropdown_value_change, names="value")

        # Restrict actions on plots
        # self.main_fig.update_layout(config=dict(displayModeBar=False))
        # [ "autoScale2d", "autoscale", "editInChartStudio", "editinchartstudio", "hoverCompareCartesian", "hovercompare", "lasso", "lasso2d", "orbitRotation", "orbitrotation", "pan", "pan2d", "pan3d", "reset", "resetCameraDefault3d", "resetCameraLastSave3d", "resetGeo", "resetSankeyGroup", "resetScale2d", "resetViewMap", "resetViewMapbox", "resetViews", "resetcameradefault", "resetcameralastsave", "resetsankeygroup", "resetscale", "resetview", "resetviews", "select", "select2d", "sendDataToCloud", "senddatatocloud", "tableRotation", "tablerotation", "toImage", "toggleHover", "toggleSpikelines", "togglehover", "togglespikelines", "toimage", "zoom", "zoom2d", "zoom3d", "zoomIn2d", "zoomInGeo", "zoomInMap", "zoomInMapbox", "zoomOut2d", "zoomOutGeo", "zoomOutMap", "zoomOutMapbox", "zoomin", "zoomout"]
        self.main_fig.update_layout(modebar=dict(remove=["select", "lasso"], orientation="v"))
        self.main_fig.layout.dragmode = 'zoom'
        if self.overview_fig:
            self.overview_fig.update_layout(modebar_remove=["toimage", "autoscale", "select", "lasso", "pan", "reset", "resetscale", "zoom", "zoomin", "zoomout"])
            self.overview_fig.layout.dragmode = False

        # Return the container
        return container
    
    def build_node_trace(self, is_overview:bool, size_by:str, color_by:str):
        if is_overview:
            hovertemplate='%{customdata[0]}<extra></extra>'
        else:
            hovertemplate=(
                'Name: %{customdata[0]}<br>' +
                'Operation: %{customdata[1]}<br>' +
                'Activations: %{customdata[2]}<br>' +
                'Parameters: %{customdata[3]}<br>' +
                'FLOPS: %{customdata[4]}<br>' +
                '<br>' +
                'args: %{customdata[5]}<br>' +
                'kwargs: %{customdata[6]}<br>' +
                '<extra></extra>'
            )
        return go.Scatter(
            x=[n.x for n in self.g.nodes], 
            y=[n.y for n in self.g.nodes], 
            mode='markers', 
            marker=self.build_marker_dict(is_overview, size_by, color_by),
            hovertemplate=hovertemplate,
            customdata=[self.node_data(n) + self.node_arg_data(n) for n in self.g.nodes],
            showlegend=False
        )
    
    def build_marker_dict(self, is_overview:bool, size_by:str, color_by:str):
        g = self.g
        if is_overview:
            if size_by=='flops':
                sizes = [self.flops_to_dot_size_overview(n.flops) for n in self.g.nodes]
            elif size_by=='params':
                sizes = [self.params_to_dot_size_overview(n.params) for n in self.g.nodes]
            else:
                raise ValueError(f'Unknown size_by: {size_by}')
        else:
            if size_by=='flops':
                sizes = [self.flops_to_dot_size(n.flops) for n in self.g.nodes]
            elif size_by=='params':
                sizes = [self.params_to_dot_size(n.params) for n in self.g.nodes]
            else:
                raise ValueError(f'Unknown size_by: {size_by}')
        
        colors = [self.get_node_color(n, color_by) for n in self.g.nodes]

        return dict(size=sizes, color=colors, colorscale=self.continuous_colorscale)

    def column_headings(self):
        total_params = sum([n.params for n in self.g.nodes])
        total_flops = sum([n.flops for n in self.g.nodes])
        return ('Name', 'Operation', 'Activations', f'Params [{self.fmt_large(total_params)}]', f'FLOPS [{self.fmt_large(total_flops)}]')

    def fmt_large(self, large_value):
        return f'{large_value:,}'.replace(',', ' ')

    def fmt_activ(self, activations):
        return f"({','.join(map(str, activations))})"

    def node_data(self, n:MavNode):
        return (n.name, n.operation, self.fmt_activ(n.activations), self.fmt_large(n.params), self.fmt_large(n.flops))

    def node_arg_data(self, n:MavNode):
        return (n.metadata.get('args',''), n.metadata.get('kwargs',''))

    def params_to_norm_val(self, params):
        """
        Obtains a logarithmically scaled value between 0 (fewest 
        parameters) and 1 (most parameters) to use for dot color 
        or size scaling, as needed.
        """
        # 
        # * Early exit: If the largest node does not even 1.007 times the 
        #   number of params than that of the smallest node, just give them 
        #   all the same value to stay clear of small denominators
        if self.params_log_ratio is None or self.params_log_ratio < 0.01: return 0.5
        v = np.clip(params, self.params_range[0], self.params_range[1])
        v_norm = (np.log2(v) - np.log2(self.params_range[0])) / self.params_log_ratio  # Scaled to between 0 and 1
        return v_norm

    def params_to_dot_size(self, params):
        dot_range = [6,18] # Plotly default size is 6
        v_norm = self.params_to_norm_val(params)
        return dot_range[0] + v_norm*(dot_range[1]-dot_range[0])

    def params_to_dot_size_overview(self, params):
        overview_dot_range = [4,10] # Plotly default size is 6
        v_norm = self.params_to_norm_val(params)
        return overview_dot_range[0] + v_norm*(overview_dot_range[1]-overview_dot_range[0])

    def flops_to_norm_val(self, flops):
        """
        Obtains a logarithmically scaled value between 0 (fewest 
        FLOPS) and 1 (most FLOPS) to use for dot color or size 
        scaling, as needed.
        """
        # * Early exit: If the largest node does not even 1.007 times the 
        #   number of FLOPS than that of the smallest node, just give them 
        #   all the same value to stay clear of small denominators
        if self.flops_log_ratio is None or self.flops_log_ratio< 0.01: return 0.5
        v = np.clip(flops, self.flops_range[0], self.flops_range[1])
        v_norm = (np.log2(v) - np.log2(self.flops_range[0])) / self.flops_log_ratio 
        return v_norm

    def flops_to_dot_size(self, flops):
        dot_range = [6,18] # Plotly default size is 6
        v_norm = self.flops_to_norm_val(flops)
        return dot_range[0] + v_norm*(dot_range[1]-dot_range[0])
    
    def flops_to_dot_size_overview(self, flops):
        overview_dot_range = [4,10] # Plotly default size is 6
        v_norm = self.flops_to_norm_val(flops)
        return overview_dot_range[0] + v_norm*(overview_dot_range[1]-overview_dot_range[0])

    def get_node_color(self, node:MavNode, color_by='operation'):
        if color_by == 'operation': 
            return node.op_color
        elif color_by == 'flops': 
            return self.flops_to_norm_val(node.flops)
        elif color_by == 'params': 
            return self.params_to_norm_val(node.params)
        else: 
            raise ValueError(f'Unknown color style: {color_by}')

    def html_scrolling_table_id(self):
        return f'scr_table_{self.unique_id}'

    def html_table_id(self):
        return f'table_{self.unique_id}'

    def html_cell_id(self, row_idx, col_idx):
        return f'cell_{self.unique_id}_{row_idx}_{col_idx}'

    def write_table_style(self):
        """
        Generates the HTML style element (<style>...</style>) 
        used to render the table
        """
        lines = []
        lines.append('<style>')
        lines.append('.highlight {background-color: #b0c0e0;}')
        lines.append('thead th {position: sticky; top:0; z-index: 1; background-color: #ebebeb;}')
        lines.append('</style>')
        return '\n'.join(lines)

    def write_row_html(self, row_idx:int, n:MavNode):
        """
        Generates an HTML row (<tr>...</tr>) containing the data
        associated with the specified node
        """
        lines = []
        lines.append('    <tr>')
        row_data = self.node_data(n)
        for col_idx, value in enumerate(row_data):
            lines.append(f'      <td id="{self.html_cell_id(row_idx, col_idx)}">{value}</td>')
        lines.append('    </tr>')
        return lines

    def write_table_html(self, g:MavGraph):
        """
        Generates an HTML row (<table>...</table>) containing the data
        associated with the specified graph
        """
        # Start of structure 
        lines = []
        lines.append(f'<table id="{self.html_table_id()}">')

        # Header
        header_cols = self.column_headings()
        lines.append('  <thead>')
        lines.append('    <tr>')
        for header_value in header_cols:
            lines.append(f'      <th>{header_value}</th>')
        lines.append('    </tr>')
        lines.append('  <thead>')
            
        # Rows
        lines.append('  <tbody>')
        for row_idx, n in enumerate(g.nodes):
            lines += self.write_row_html(row_idx, n)

        # End of structure
        lines.append('  </tbody>')
        lines.append('</table>')
        return '\n'.join(lines)
    
    def fit_range(self, target_range, full_range):
        full_size = max(full_range) - min(full_range)
        result_size = max(target_range) - min(target_range)
        result_start = min(target_range)
        if result_size > full_size: result_size = full_size
        if result_start < min(full_range): result_start = min(full_range)
        if result_start > max(full_range)-result_size: result_start = max(full_range)-result_size
        if full_range[0] < full_range[1]:
            return [result_start, result_start+result_size]
        else:
            return [result_start+result_size, result_start]

    def select_node(self, idx):
        # Pan the main panel to have the clicked point in the middle
        # * Clip near the edges to avoid scrolling past the beginning or end
        # * This will also update the overview rect "on_main_panel_pan_zoom"
        self.pan_to_center(self.g.nodes[idx].x, self.g.nodes[idx].y)

        # Update selected marker
        node = self.g.nodes[idx]
        if self.sel_marker_trace is not None:
            self.sel_marker_trace.update(
                x=[node.x], y=[node.y], 
                marker_size=[self.params_to_dot_size(node.params)]
            )
        
        # Scroll the table to the clicked module and highlight the selected node
        # * Do this after panning the main panel, because it might also trigger
        #   a scroll action
        if self.table_widget:
            num_cols = len(self.column_headings())
            js_lines = []
            js_lines.append(f'const table = document.getElementById("{self.html_table_id()}");')
            js_lines.append('const highlightedCells = table.querySelectorAll(".highlight");')
            js_lines.append('highlightedCells.forEach(cell => cell.classList.remove("highlight"));')
            for ci in range(num_cols):
                js_lines.append(f'document.getElementById("{self.html_cell_id(idx,ci)}").classList.add("highlight");')
            js_lines.append(f'document.getElementById("{self.html_cell_id(idx,0)}").scrollIntoView({{behavior:"smooth", block:"center", inline:"nearest"}});')
            js = '\n'.join(js_lines)
            display(Javascript(js))

    def pan_to_center(self, x, y):
        # Pan the main panel
        # * This will also trigger updates for the overview rect and slider via on_main_panel_pan_zoom
        w = self.main_fig.layout.xaxis.range[1] - self.main_fig.layout.xaxis.range[0]
        h = self.main_fig.layout.yaxis.range[0] - self.main_fig.layout.yaxis.range[1]  # Note: order is inverted
        x_range = self.fit_range([x-w/2, x+w/2], self.full_x_range)
        y_range = self.fit_range([y+h/2, y-h/2], self.full_y_range)
        self.main_fig.update_layout(xaxis=dict(range=x_range), yaxis=dict(range=y_range))  

    def autoscroll_table(self):
        # Scroll the table
        # * Find the lowest-index node in view and scroll such that it is at the top of the table
        # * Note the order inversion in y_range
        if self.table_widget:
            x_range = self.main_fig.layout.xaxis.range
            y_range = self.main_fig.layout.yaxis.range
            idxs = [n._idx for n in self.g.nodes if n.x >= x_range[0] and n.x <= x_range[1] and n.y >= y_range[1] and n.y <= y_range[0]]
            if idxs:
                idx = min(idxs)
                # We need to subtract an offset and treat the idx==0 case separately, because
                # "scrollIntoView" with block:"start" scrolls the referenced row behind the
                # header row
                if idx<=0: 
                    js = f'document.getElementById("{self.html_scrolling_table_id()}").scrollTo({{behavior:"smooth", top:0, left:0}});'
                else:
                    idx-= 1  # Compensate for offset caused by making header sticky
                    js = f'document.getElementById("{self.html_cell_id(idx,0)}").scrollIntoView({{behavior:"smooth", block:"start", inline:"nearest"}});'
                display(Javascript(js))

    def on_main_panel_click(self, trace, points:cb.Points, selector):
        if not points.point_inds: return        
        idx = points.point_inds[0]
        self.select_node(idx)

    def on_overview_panel_click(self, trace, points:cb.Points, selector):
        if not points.point_inds: return
        idx = points.point_inds[0]
        self.pan_to_center(self.g.nodes[idx].x, self.g.nodes[idx].y)

    def on_slider_value_change(self, value_dict):
        # Skip if this was programmatically updated to avoid infinite recursion. Most
        # component versions may not require this but it's better to be safe
        if self.updating_slider: return

        # * Take the negative of the y_slider values to compensate for the fact that
        #   these values have been negated to invert the direction of the slider
        self.main_fig.update_layout(
            yaxis=dict(range=[-self.slider_widget.value[0], -self.slider_widget.value[1]])
        )

    def on_dropdown_value_change(self, value_dict):
        size_by, color_by = self.dropdown_widget.value
        if self.node_marker_trace:
            self.node_marker_trace.update(marker=self.build_marker_dict(False, size_by, color_by))
        if self.overview_marker_trace:
            self.overview_marker_trace.update(marker=self.build_marker_dict(True, size_by, color_by))

    def on_main_panel_pan_zoom(self, layout, x_range, y_range):
        # Update rectangle on overview panel
        if self.overview_fig:
            x0, x1 = x_range
            y0, y1 = y_range
            self.overview_rect_trace.update(
                x=[x0, x0, x1, x1, x0], 
                y=[y0, y1, y1, y0, y0]
            )

        # Update the slider value
        # * Set the state variable to avoid infinite recursion in case some component 
        #   versions require this
        # * Take the negative of the widget value (see above)
        if self.slider_widget:
            self.updating_slider = True
            self.slider_widget.value = [-y_range[0], -y_range[1]]
            self.updating_slider = False

        # Auto-scroll the table
        self.autoscroll_table()

    def get_connection_coords(self, c:MavConnection):
        if use_straight_connection(c, self.g):
            return [c.from_node.x, c.to_node.x], [c.from_node.y, c.to_node.y]
        else:
            # The curved lines display nicely at some levels of zoom, but look awkward
            # at others, especially when the vertical dimension is zoomed out.
            # The segmented lines are more consistent and use fewer points.
            offset = c.offset if c.offset is not None else 0.4
            return segmented_line_coords(c.from_node.x, c.from_node.y, c.to_node.y, offset)
