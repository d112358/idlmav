from idlmav.mavtypes import MavGraph, MavNode, MavConnection
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReleaseViewer:
    """
    This viewer avoids `go.FigureWidget` and `ipywidgets`, making it 
    suitable for use when releasing a model, since users browsing 
    through models on GitHub without any running Jupyter kernel or 
    dynamic components will still be able to see all output produced 
    by this viewer (e.g. using nbviewer).

    Available interactions and limitations:
    * Hover interactions are available, but no click events
    * An optional table is available, but no interaction between the 
      graph and the table is available.
    * An optional scroll bar is available, but unfortunately only a 
      horizontal one
    """

    # TODO: Add drop-down boxes for node coloring and sizing: https://plotly.com/python/dropdowns/

    def __init__(self, g:MavGraph):
        self.g = g
        self.color_style = 'operation'  # ['operation','params','flops']

        # Derived parameters
        self.min_x = min([n.x for n in g.nodes])
        self.max_x = max([n.x for n in g.nodes])
        self.graph_num_cols = self.max_x - self.min_x + 1

        all_params = [n.params for n in g.nodes]
        all_flops = [n.flops for n in g.nodes]
        pos_params = [v for v in all_params if v > 0]
        pos_flops = [v for v in all_flops if v > 0]        
        self.params_range = [min(pos_params), max(pos_params)] if pos_params else [0,0]
        self.flops_range = [min(pos_flops), max(pos_flops)] if pos_flops else [0,0]
        self.params_log_ratio = np.log2(self.params_range[1]) - np.log2(self.params_range[0])
        self.flops_log_ratio = np.log2(self.flops_range[1]) - np.log2(self.flops_range[0])

    def draw(self, add_table:bool=True, add_slider:bool=False, num_levels_displayed:float=10, *args, **kwargs):
        g = self.g
    
        # Create figure, possibly with subplots
        num_subplots = 1 
        subplot_specs=[[{"type": "scatter"}]]
        column_widths = [self.graph_num_cols]
        if add_table:
            table_col_scale_factor = 1.8  # If set to 1, a table column takes up the same width as a column of nodes in the graph
            num_subplots += 1
            subplot_specs[0] += [{"type": "table"}]
            column_widths.append(table_col_scale_factor*len(self.column_headings()))
        fig = make_subplots(rows=1, cols=num_subplots, vertical_spacing=0.03, specs=subplot_specs,
                            column_widths=column_widths)

        # Draw nodes
        scatter_trace = go.Scatter(
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
        fig.add_trace(scatter_trace, row=1, col=1)
        fig.update_xaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        fig.update_yaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])

        # Display direction
        in_level = g.in_nodes[0].y
        out_level = g.out_nodes[0].y
        fig.update_yaxes(range=[out_level+0.5, in_level-0.5])

        # Add connections
        for c in g.connections:
            x_coords, y_coords = self.get_connection_coords(c)
            line_trace = go.Scatter(
                x=x_coords, y=y_coords, mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False
            )
            fig.add_trace(line_trace, row=1, col=1)

        # Add table if selected
        if add_table:
            table_trace = go.Table(
                header=dict(
                    values=['Name', 'Operation', 'Activations', 'Params', 'FLOPS'],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[[n.name for n in g.nodes], 
                            [n.operation for n in g.nodes], 
                            [n.activations for n in g.nodes], 
                            [n.params for n in g.nodes], 
                            [n.flops for n in g.nodes]],
                    align = "left")
            )
            fig.add_trace(table_trace, row=1, col=2)
        
        # Add slider if selected
        if add_slider:
            steps = []
            for ilvl in np.arange(in_level, out_level-num_levels_displayed+1, 0.1):
                step = dict(
                    method="relayout",
                    args=[{"yaxis": {"range": [ilvl+num_levels_displayed-0.5, ilvl-0.5],
                                     "showgrid":False, "zeroline":False, "tickmode":'array', "tickvals":[]}}],
                    label="",
                )
                steps.append(step)

            slider = dict(
                active=0,
                currentvalue={"visible": False},
                pad={"t": 50},
                steps=steps,
                ticklen=0,
                tickwidth=0
            )

            fig.update_layout(
                sliders=[slider]
            )
            fig.update_yaxes(range=[in_level+num_levels_displayed-0.5, in_level-0.5])

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig
    
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

def use_straight_connection(c:MavConnection, g:MavGraph):
    n0, n1 = c.from_node, c.to_node
    x0, y0, x1, y1 = n0.x, n0.y, n1.x, n1.y
    if x0 != x1: return True  # Use straght lines unless vertical and obstructed by another node
    nodes_on_line = [n for n in g.nodes if n.x == x0]  # First just perform one check on all nodes
    nodes_on_segment = [n for n in nodes_on_line if n.y > y0 and n.y < y1]  # Perform other 2 checks on subset of nodes
    return False if nodes_on_segment else True
        
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

def curved_line_coords(x01, y0, y1):
    """
    `curved_line_coords` return the x and y coordinates for a 
    curved line that connects two nodes on the same horizontal
    coordinate `x01` and different vertical coordinates `y0` and 
    `y1`. This is an alternative to `segmented_line_coords`
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