from .mavtypes import MavNode, MavGraph, MavConnection
import time
import numpy as np
import plotly.graph_objects as go
import plotly.callbacks as cb
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
    # TODO: Add drop-down boxes for node coloring and sizing: https://plotly.com/python/dropdowns/

    def __init__(self, g:MavGraph):
        self.g = g

        # Panels and widgets
        self.main_panel     : widgets.Box     = None
        self.table_panel    : widgets.Box     = None
        self.overview_panel : widgets.Box     = None
        self.slider_panel   : widgets.Box     = None
        self.main_fig       : go.FigureWidget = None
        self.table_widget   : widgets.Output  = None
        self.overview_fig   : go.FigureWidget = None
        self.slider_widget  : widgets.FloatRangeSlider = None

        # Annotations
        self.overview_rect_idx  : int = None
        self.sel_marker_idx     : int = None

        # Derived parameters
        self.in_level = g.in_nodes[0].y
        self.out_level = g.out_nodes[0].y
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
        self.params_log_ratio = np.log2(self.params_range[1]) - np.log2(self.params_range[0])
        self.flops_log_ratio = np.log2(self.flops_range[1]) - np.log2(self.flops_range[0])

        # State variables
        self.color_style = 'operation'  # ['operation','params','flops']
        self.unique_id = f'{id(self)}_{int(time.time() * 1000)}'
        self.updating_slider = False

    def render(self, add_table:bool=True, add_slider:bool=True, add_overview:bool=False, num_levels_displayed:float=10, height_px=400, *args, **kwargs) -> widgets.Box:
        # Setup parameters
        g = self.g
        initial_y_range = self.fit_range([self.in_level+num_levels_displayed-0.5, self.in_level-0.5], self.full_y_range)
        initial_x_range = self.full_x_range

        # Create a new unique ID every time this is called        
        self.unique_id = f'{id(self)}_{int(time.time() * 1000)}'
    
        # Create the main panel
        main_panel_layout = widgets.Layout(flex = '0 1 auto', margin='0px', padding='0px', overflow='hidden')
        main_fig_layout = go.Layout(
            width=max((self.graph_num_cols*100, 180)), height=height_px,
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
            hovertemplate='<extra></extra>', showlegend=False
        )
        self.main_fig.add_trace(sel_marker)
        self.sel_marker_idx = len(self.main_fig.data)-1

        # Add connections lines between the nodes
        # * Use a single trace with `None` values separating different lines
        # * Using a separate trace for every line cause a blank display 
        #   on Colab
        # * Separate traces may also negatively impact reactivity, e.g. 
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
            showlegend=False
        )
        self.main_fig.add_trace(line_trace)

        # Add the node markers
        node_trace = go.Scatter(
            x=[n.x for n in g.nodes], 
            y=[n.y for n in g.nodes], 
            mode='markers', 
            marker=dict(
                size=[self.params_to_dot_size(n.params) for n in g.nodes],
                color=[self.get_node_color(n) for n in g.nodes],
                colorscale='Bluered'
            ),
            hovertemplate=(
                'Name: %{customdata[0]}<br>' +
                'Operation: %{customdata[1]}<br>' +
                'Activations: %{customdata[2]}<br>' +
                'Parameters: %{customdata[3]}<br>' +
                'FLOPS: %{customdata[4]}<br>' +
                '<extra></extra>'
            ),
            customdata=[self.node_data(n) for n in g.nodes],
            showlegend=False
        )
        self.main_fig.add_trace(node_trace)
        node_trace_idx = len(self.main_fig.data)-1

        # Add table if selected
        if add_table:
            table_panel_layout = widgets.Layout(flex='0 0 auto', margin='0px', padding='0px', overflow='visible')
            table_style = self.write_table_style()
            table_html = self.write_table_html(g)
            scrolling_table_html = f'<div id="{self.html_scrolling_table_id()}" style="height: {height_px}px; overflow: auto; width: fit-content">{table_html}</div>'
            self.table_widget = widgets.Output()
            with self.table_widget:
                display(HTML(table_style))
                display(HTML(scrolling_table_html))
            self.table_panel = widgets.Box(children=[self.table_widget], layout=table_panel_layout)
            panels.append(self.table_panel)
            
        # Add overview window if selected
        overview_trace_idx = None
        if add_overview:
            # Overview panel
            overview_panel_layout = widgets.Layout(flex = '0 0 auto', margin='0px', padding='0px', overflow='hidden')
            overview_fig_layout = go.Layout(
                width=max((self.graph_num_cols*15, 45)),
                height=height_px,
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
            overview_nodes_trace = go.Scatter(
                x=[n.x for n in g.nodes], 
                y=[n.y for n in g.nodes], 
                mode='markers', 
                marker=dict(
                    size=[self.params_to_dot_size_overview(n.params) for n in g.nodes],
                    color=[self.get_node_color(n) for n in g.nodes],
                    colorscale='Bluered'
                ),
                hovertemplate='%{customdata[0]}<extra></extra>',
                customdata=[self.node_data(n) for n in g.nodes],
                showlegend=False
            )
            self.overview_fig.add_trace(overview_nodes_trace)
            overview_trace_idx = len(self.overview_fig.data)-1

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
                hovertemplate='<extra></extra>',
                showlegend=False
            )
            self.overview_fig.add_trace(rect_trace)
            self.overview_rect_idx = len(self.overview_fig.data)-1

        # Add slider if selected
        # * Use negative values everywhere, because ipywidgets does not support
        #   inverting the direction of vertical sliders
        if add_slider:
            slider_panel_layout = widgets.Layout(flex = '0 0 auto', margin='0px', padding='0px', overflow='visible')
            self.slider_widget = widgets.FloatRangeSlider(
                value=[-initial_y_range[0], -initial_y_range[1]], min=-self.full_y_range[0], max=-self.full_y_range[1], 
                step=0.01, description='', orientation='vertical', continuous_update=True,
                layout=widgets.Layout(height=f'{height_px}px')
            )
            self.slider_widget.readout = False  # For some reason it does not seem to work if set during construction
            self.slider_panel = widgets.Box(children=[self.slider_widget], layout=slider_panel_layout)
            panels.insert(0, self.slider_panel)

        # Create container for all panels
        # * To be displayed in Notebook using `display`
        container_layout = widgets.Layout(
            width='100%',
            margin='0px', padding='0px')
        container = widgets.HBox(panels, layout=container_layout)

        # Set up event handlers        
        self.main_fig.data[node_trace_idx].on_click(self.on_main_panel_click)
        self.main_fig.layout.on_change(self.on_main_panel_pan_zoom, 'xaxis.range', 'yaxis.range')
        if self.overview_fig:
            self.overview_fig.data[overview_trace_idx].on_click(self.on_overview_panel_click)
        if self.slider_widget:
            self.slider_widget.observe(self.on_slider_value_change, names="value")

        # Restrict actions on plots
        # self.main_fig.update_layout(config=dict(displayModeBar=False))
        # [ "autoScale2d", "autoscale", "editInChartStudio", "editinchartstudio", "hoverCompareCartesian", "hovercompare", "lasso", "lasso2d", "orbitRotation", "orbitrotation", "pan", "pan2d", "pan3d", "reset", "resetCameraDefault3d", "resetCameraLastSave3d", "resetGeo", "resetSankeyGroup", "resetScale2d", "resetViewMap", "resetViewMapbox", "resetViews", "resetcameradefault", "resetcameralastsave", "resetsankeygroup", "resetscale", "resetview", "resetviews", "select", "select2d", "sendDataToCloud", "senddatatocloud", "tableRotation", "tablerotation", "toImage", "toggleHover", "toggleSpikelines", "togglehover", "togglespikelines", "toimage", "zoom", "zoom2d", "zoom3d", "zoomIn2d", "zoomInGeo", "zoomInMap", "zoomInMapbox", "zoomOut2d", "zoomOutGeo", "zoomOutMap", "zoomOutMapbox", "zoomin", "zoomout"]
        self.main_fig.update_layout(modebar_remove=["toimage", "resetscale", "select", "lasso", "reset"])
        self.main_fig.layout.dragmode = 'zoom'
        if self.overview_fig:
            self.overview_fig.update_layout(modebar_remove=["toimage", "autoscale", "select", "lasso", "pan", "reset", "resetscale", "zoom", "zoomin", "zoomout"])
            self.overview_fig.layout.dragmode = False

        # Return the container
        return container
    
    def column_headings(self):
        return ('Name', 'Operation', 'Activations', 'Params', 'FLOPS')

    def node_data(self, n:MavNode):
        return (n.name, n.operation, n.activations, n.params, n.flops)

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
        if self.params_log_ratio < 0.01: return 0.5
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
        if self.flops_log_ratio < 0.01: return 0.5
        v = np.clip(flops, self.flops_range[0], self.flops_range[1])
        v_norm = (np.log2(v) - np.log2(self.params_range[0])) / self.flops_log_ratio 
        return v_norm

    def flops_to_dot_size(self, flops):
        dot_range = [6,18] # Plotly default size is 6
        v_norm = self.flops_to_norm_val(flops)
        return dot_range[0] + v_norm*(dot_range[1]-dot_range[0])
    
    def get_node_color(self, node:MavNode):
        if self.color_style == 'operation': 
            return node.op_color
        elif self.color_style == 'flops': 
            return self.flops_to_norm_val(node.flops)
        else: 
            return self.params_to_norm_val(node.params)

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
        if self.sel_marker_idx is not None:
            self.main_fig.data[self.sel_marker_idx].update(
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

    def on_main_panel_pan_zoom(self, layout, x_range, y_range):
        # Update rectangle on overview panel
        if self.overview_fig:
            x0, x1 = x_range
            y0, y1 = y_range
            self.overview_fig.data[self.overview_rect_idx].update(
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
        # TODO: Update all connection coords functions when supporting horizontal main direction
        # TODO: Move all connection coords functions to base class to share with other viewers
        if use_straight_connection(c, self.g):
            return [c.from_node.x, c.to_node.x], [c.from_node.y, c.to_node.y]
        else:
            # The curved lines display nicely at some levels of zoom, but look awkward
            # at others, especially when the vertical dimension is zoomed out.
            # The segmented lines are more consistent and use fewer points.
            return segmented_line_coords(c.from_node.x, c.from_node.y, c.to_node.y)
        
def segmented_line_coords(x01, y0, y1):
    """
    `segmented_line_coords` returns the x and y coordinates for a
    segmented line that connects two nodes on the same horizontal
    coordinate `x01` and different vertical coordinates `y0` and 
    `y1`. This is an alternative to `curved_line_coords`.
    """
    r=0.2
    ymin, ymax = min([y0,y1]), max([y0,y1])
    x = [x01,  x01+2*r,  x01+2*r,  x01]
    y = [ymin, ymin+2*r, ymax-2*r, ymax]
    return x, y

def use_straight_connection(c:MavConnection, g:MavGraph):
    n0, n1 = c.from_node, c.to_node
    x0, y0, x1, y1 = n0.x, n0.y, n1.x, n1.y
    if x0 != x1: return True  # Use straght lines unless vertical and obstructed by another node
    nodes_on_line = [n for n in g.nodes if n.x == x0]  # First just perform one check on all nodes
    nodes_on_segment = [n for n in nodes_on_line if n.y > y0 and n.y < y1]  # Perform other 2 checks on subset of nodes
    return False if nodes_on_segment else True
        
def curved_line_coords(x01, y0, y1):
    """
    `curved_line_coords` return the x and y coordinates for a 
    curved line that connects two nodes on the same horizontal
    coordinate `x01` and different vertical coordinates `y0` and `y1`
    """
    r = 0.2
    ymin, ymax = min([y0,y1]), max([y0,y1])
    x, y = [], []
    append_arc_coords(x, y, x01+r, ymin, r, 3, True)
    append_arc_coords(x, y, x01+r, ymin+2*r, r, 1, False)
    append_arc_coords(x, y, x01+r, ymax-2*r, r, 4, False)
    append_arc_coords(x, y, x01+r, ymax, r, 2, True)
    return x, y

def append_arc_coords(x, y, cx, cy, r, quadrant, ccw:bool, num_points:int=20):
    """
    `append_arc_coords` appends x and y-coordinates for the 
    specified arc to the existing lists of coordinates `x` and `y`.

    The appended arc is always a quarter-circle and is defined by
    the centre `cx`, `cy` and radius `r` of the circle, the 
    quadrant `quadrant` for which to return coordinates, the
    boolean flag `ccw` specifying whether to sort the coordinates
    in counter-clockwise order and the number of points 
    `num_points` into which the arc should be divided.

    Quadrants are numbered as follows:
        2=TopLeft       |    1=TopRight
        ----------------------------------
        3=BottomLeft    |    4=BottomRight
    """
    if ccw:
        t = np.linspace(np.pi/2*(quadrant-1), np.pi/2*quadrant, num_points)
    else:
        t = np.linspace(np.pi/2*quadrant, np.pi/2*(quadrant-1), num_points)
    xdata = cx + r*np.cos(t)
    ydata = cy - r*np.sin(t)  # Negative because y-axis is inverted on graph
    x += list(xdata)
    y += list(ydata)

__all__ = ["WidgetRenderer"]