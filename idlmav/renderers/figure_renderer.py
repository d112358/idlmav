from ..mavtypes import MavGraph, MavNode, MavConnection
from .renderer_utils import use_straight_connection, segmented_line_coords
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FigureRenderer:
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

        # Derived parameters
        self.in_level = min([n.y for n in g.in_nodes])
        self.out_level = max([n.y for n in g.out_nodes])
        self.min_x = min([n.x for n in g.nodes])
        self.max_x = max([n.x for n in g.nodes])
        self.graph_num_cols = self.max_x - self.min_x + 1

        all_params = [n.params for n in g.nodes]
        all_flops = [n.flops for n in g.nodes]
        pos_params = [v for v in all_params if v > 0]
        pos_flops = [v for v in all_flops if v > 0]        
        self.params_range = [min(pos_params), max(pos_params)] if pos_params else [0,0]
        self.flops_range = [min(pos_flops), max(pos_flops)] if pos_flops else [0,0]
        self.params_log_ratio = np.log2(self.params_range[1]) - np.log2(self.params_range[0]) if pos_params else None
        self.flops_log_ratio = np.log2(self.flops_range[1]) - np.log2(self.flops_range[0]) if pos_flops else None

        # Subplot and trace indices
        self.fig:go.Figure          = None
        self.overview_sp_col:int    = None  # 1-based
        self.main_sp_col:int        = None  # 1-based
        self.table_sp_col:int       = None  # 1-based
        self.node_trace_idx:int     = None
        self.overview_trace_idx:int = None

    def render(self, add_table:bool=True, add_slider:bool=False, add_overview:bool=False, num_levels_displayed:float=10, *args, **kwargs):
        g = self.g
        margin = 0.25
    
        # Create figure, possibly with subplots
        num_subplots = 1 
        subplot_specs=[[{"type": "scatter"}]]
        column_widths = [self.graph_num_cols]
        self.main_sp_col = 1
        if add_table:
            table_col_scale_factor = 1.8  # If set to 1, a table column takes up the same width as a column of nodes in the graph
            num_subplots += 1
            subplot_specs[0] += [{"type": "table"}]
            column_widths.append(table_col_scale_factor*len(self.column_headings()))
            self.table_sp_col = 2
        if add_overview:
            num_subplots += 1
            subplot_specs[0].insert(0, {"type": "scatter"})
            column_widths.insert(0, self.graph_num_cols / 3)
            self.overview_sp_col = 1
            self.main_sp_col += 1
            if self.table_sp_col is not None: self.table_sp_col += 1
        self.fig = make_subplots(rows=1, cols=num_subplots, vertical_spacing=0.03, specs=subplot_specs,
                            column_widths=column_widths)

        # Draw connections lines between the nodes
        # * Use a single trace with `None` values separating different lines
        # * Separate traces may negatively impact responsiveness, e.g. to pan & zoom actions
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
        self.fig.add_trace(line_trace, row=1, col=self.main_sp_col)
        if add_overview: self.fig.add_trace(line_trace, row=1, col=self.overview_sp_col)

        # Draw nodes
        node_trace = self.build_node_trace(False, 'params', 'operation')
        self.fig.add_trace(node_trace, row=1, col=self.main_sp_col)
        self.node_trace_idx = len(self.fig.data)-1
        if add_overview:
            overview_trace = self.build_node_trace(True, 'params', 'operation')
            self.fig.add_trace(overview_trace, row=1, col=self.overview_sp_col)
            self.overview_trace_idx = len(self.fig.data)-1

        # Draw overview shape
        if add_overview:
            self.fig.add_shape(type="rect",
                xref="x", yref="y",
                x0=self.min_x-margin, x1=self.max_x+margin, y0=self.in_level-margin, y1=self.in_level-margin+num_levels_displayed,
                line=dict(color="#000000"),
                row=1, col=self.overview_sp_col
            )

        # Update layout and display direction
        self.fig.update_xaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        self.fig.update_yaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        self.update_range(self.main_sp_col, [self.min_x-margin, self.max_x+margin], [self.out_level+margin, self.in_level-margin])
        if add_overview: self.update_range(self.overview_sp_col, [self.min_x-margin*2, self.max_x+margin*2], [self.out_level+margin*2, self.in_level-margin*2])

        # Add table if selected
        if add_table:
            table_trace = go.Table(
                header=dict(
                    values=self.column_headings(),
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[[n.name for n in g.nodes], 
                            [n.operation for n in g.nodes], 
                            [self.fmt_activ(n.activations) for n in g.nodes], 
                            [self.fmt_large(n.params) for n in g.nodes], 
                            [self.fmt_large(n.flops) for n in g.nodes]],
                    align = "left")
            )
            self.fig.add_trace(table_trace, row=1, col=self.table_sp_col)
        
        # Add styling dropdown menu
        self.fig.update_layout(updatemenus=[self.build_styling_menu(pad_t=8)])

        # Add slider if selected
        if add_slider:
            total_levels = self.out_level - self.in_level + margin*2
            num_slider_steps = int(total_levels / num_levels_displayed * 3.5)
            self.fig.update_layout(sliders=[self.build_overview_slider(pad_t=48, margin=margin, num_levels_displayed=num_levels_displayed, num_steps=num_slider_steps)])
            self.update_range(self.main_sp_col, [self.min_x-margin, self.max_x+margin], [self.in_level+num_levels_displayed-margin, self.in_level-margin])

        # Update margin and modebar buttons
        self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        self.fig.update_layout(modebar=dict(remove=["select", "lasso"], orientation="v"))
        return self.fig
    
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
                sizes = [self.flops_to_dot_size_overview(n.params) for n in self.g.nodes]
            elif size_by=='params':
                sizes = [self.params_to_dot_size_overview(n.params) for n in self.g.nodes]
            else:
                raise ValueError(f'Unknown size_by: {size_by}')
        else:
            if size_by=='flops':
                sizes = [self.flops_to_dot_size(n.params) for n in self.g.nodes]
            elif size_by=='params':
                sizes = [self.params_to_dot_size(n.params) for n in self.g.nodes]
            else:
                raise ValueError(f'Unknown size_by: {size_by}')
        
        colors = [self.get_node_color(n, color_by) for n in self.g.nodes]

        return dict(size=sizes, color=colors, colorscale='Bluered')

    def build_menu_button(self, size_by:str, color_by:str):
        size_color_labels = dict(operation='operation', params='params', flops='FLOPS')
        marker_list = [{}]*len(self.fig.data)
        if self.node_trace_idx is not None: marker_list[self.node_trace_idx] = self.build_marker_dict(False, size_by, color_by)
        if self.overview_trace_idx is not None: marker_list[self.overview_trace_idx] = self.build_marker_dict(True, size_by, color_by)
        return dict(
            args=[dict(marker=marker_list)],
            label=f'Size by {size_color_labels[size_by]}, color by {size_color_labels[color_by]}',
            method="restyle"
        )

    def build_styling_menu(self, pad_t=0):
        size_color_options = [('params','operation'),
                              ('flops','operation'),
                              ('params','flops'),
                              ('flops','params')]
        menu_buttons = [self.build_menu_button(size_by, color_by) for (size_by, color_by) in size_color_options]
        return dict(buttons=menu_buttons, showactive=True, direction="up",
                    pad=dict(l=0, r=0, t=pad_t, b=0),
                    x=0, xanchor="left",
                    y=0, yanchor="top")

    def build_overview_slider(self, pad_t=0, margin=0.25, num_levels_displayed=2, num_steps=20):
        steps = []
        yaxis_varname = self.ax_var_name('yaxis', self.main_sp_col)
        for i in np.linspace(self.in_level-margin, self.out_level+margin-num_levels_displayed, num_steps):
            step = dict(
                method="relayout",
                args=[dict({
                    "shapes":[dict(type="rect", xref="x", yref="y",line=dict(color="#000000"),
                        x0=self.min_x-margin, y0=i, x1=self.max_x+margin, y1=i+num_levels_displayed,
                    )],
                    yaxis_varname:dict(range=[i+num_levels_displayed, i], showgrid=False, zeroline=False, tickmode='array', tickvals=[]),
                })],
                label="",
            )
            steps.append(step)

        return dict(
                steps=steps, active=0,
                currentvalue=dict(visible=False),
                pad=dict(t=pad_t, l=0, b=0, r=0),
                x=0, xanchor="left",
                y=0, yanchor="top",
                ticklen=0, tickwidth=0
            )

    def ax_var_name(self, var_name, sp_col):
        """Ensure that the correct subplot is targeted, e.g. 'xaxis' -> 'xaxis2'"""
        return var_name if sp_col <= 1 else f"{var_name}{sp_col}"
    
    def update_range(self, sp_col, x_range, y_range):
        """Convenience method to update the axis ranges of a specific subplot"""
        xaxis_name = self.ax_var_name('xaxis', sp_col)
        yaxis_name = self.ax_var_name('yaxis', sp_col)
        kwargs = {xaxis_name: dict(range=x_range),
                  yaxis_name: dict(range=y_range)}
        self.fig.update_layout(**kwargs)

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
        return (n.metadata['args'], n.metadata['kwargs'])

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
        if self.flops_log_ratio is None or self.flops_log_ratio < 0.01: return 0.5
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

    def get_node_color(self, node:MavNode, color_style='operation'):
        # Allow passing in color_style externally to generate marker colors
        # for Plotly custom dropdown menus
        if color_style == 'operation': 
            return node.op_color
        elif color_style == 'flops': 
            return self.flops_to_norm_val(node.flops)
        elif color_style == 'params': 
            return self.params_to_norm_val(node.flops)
        else: 
            raise ValueError(f'Unknown color style: {color_style}')
        
    def get_connection_coords(self, c:MavConnection):
        if use_straight_connection(c, self.g):
            return [c.from_node.x, c.to_node.x], [c.from_node.y, c.to_node.y]
        else:
            # The curved lines display nicely at some levels of zoom, but look awkward
            # at others, especially when the vertical dimension is zoomed out.
            # The segmented lines are more consistent and use fewer points.
            offset = c.offset if c.offset is not None else 0.4
            return segmented_line_coords(c.from_node.x, c.from_node.y, c.to_node.y, offset)
