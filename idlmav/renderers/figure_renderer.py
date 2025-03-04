from ..mavoptions import RenderOptions
from ..mavtypes import MavGraph, MavNode, MavConnection
from .renderer_utils import use_straight_connection, segmented_line_coords
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

class FigureRenderer:
    """
    Class that renders a graph as a `go.Figure` object

    This renderer avoids `go.FigureWidget` and `ipywidgets`, making it 
    portable so that users browsing through models without any running 
    Jupyter kernel or backend will still be able to see all output produced 
    by this renderer (e.g. using nbviewer).

    Available interactions and limitations:
    * Hover interactions are available, but no click events
    * Standard pan and zoom controls provided by Plotly are available
    * An optional table is available, but no interaction between the 
      graph and the table.
    * An optional scroll bar is available, but unfortunately only a 
      horizontal one
    * An optional overview panel is available to show the zoomed-out
      model and highlight the area into which the main panel is zoomed.
      This panel is not clickable and is only updated from the optional 
      slider, not from the standard Plotly pan and zoom controls
    * A dropdown menu is available to select different criteria for node
      marker sizing and coloring, e.g. by operation, by number of 
      parameters, by number of FLOPS
    """

    def __init__(self, g:MavGraph):
        self.g = g

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

    def render(self, opts:RenderOptions=RenderOptions(), **kwargs):
        """
        Renders the graph received during construction as a `go.Figure` object

        Keyword arguments may be passed either via a `RenderOptions` object or
        as-is. Using a `RenderOptions` object provides better intellisense, 
        but plain keyword arguments results in more concise code.

        The following two lines are equivalent:
        ```
        fig = FigureRenderer(graph).render(RenderOptions(add_overview=True, height_px=500))  
        fig = FigureRenderer(graph).render(add_overview=True, height_px=500)  
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
        `plotly.graph_objects.Figure` object that can be displayed using `.show()`
        or exported using `.write_html()`
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)

        # Setup parameters
        g = self.g
        self.continuous_colorscale = opts.continuous_colorscale
        self.size_color_idx = opts.size_color_idx
        x_margin = 0.5  # Higher to cater for skip connections with offsets to avoid overlapping
        y_margin = 0.25
    
        # Create figure, possibly with subplots
        num_subplots = 1 
        subplot_specs=[[{"type": "scatter"}]]
        column_widths = [self.graph_num_cols]
        self.main_sp_col = 1
        if opts.add_table:
            table_col_scale_factor = 1.8  # If set to 1, a table column takes up the same width as a column of nodes in the graph
            num_subplots += 1
            subplot_specs[0] += [{"type": "table"}]
            column_widths.append(table_col_scale_factor*len(self.column_headings()))
            self.table_sp_col = 2
        if opts.add_overview:
            num_subplots += 1
            subplot_specs[0].insert(0, {"type": "scatter"})
            column_widths.insert(0, self.graph_num_cols / 3)
            self.overview_sp_col = 1
            self.main_sp_col += 1
            if self.table_sp_col is not None: self.table_sp_col += 1
        self.fig = make_subplots(rows=1, cols=num_subplots, horizontal_spacing=0.01, specs=subplot_specs,
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
        if opts.add_overview: self.fig.add_trace(line_trace, row=1, col=self.overview_sp_col)

        # Draw nodes
        if self.size_color_idx is None:
            total_non_input_params = sum([n.params for n in g.nodes if n not in g.in_nodes])
            total_non_input_flops = sum([n.flops for n in g.nodes if n not in g.in_nodes])
            initially_size_by_flops = total_non_input_params == 0 and total_non_input_flops > 0
            self.size_color_idx = 1 if initially_size_by_flops else 0
        size_by, color_by = self.size_color_options[self.size_color_idx]
        node_trace = self.build_node_trace(False, size_by, color_by)
        self.fig.add_trace(node_trace, row=1, col=self.main_sp_col)
        self.node_trace_idx = len(self.fig.data)-1
        if opts.add_overview:
            overview_trace = self.build_node_trace(True, size_by, color_by)
            self.fig.add_trace(overview_trace, row=1, col=self.overview_sp_col)
            self.overview_trace_idx = len(self.fig.data)-1

        # Draw overview shape
        if opts.add_overview:
            self.fig.add_shape(type="rect",
                xref="x", yref="y",
                x0=self.min_x-x_margin, x1=self.max_x+x_margin, y0=self.in_level-y_margin, y1=self.in_level-y_margin+opts.num_levels_displayed,
                line=dict(color='#2A3F5F', width=1), fillcolor='rgba(42,63,96,0.2)',
                row=1, col=self.overview_sp_col
            )

        # Update layout and display direction
        self.fig.update_xaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        self.fig.update_yaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        self.update_range(self.main_sp_col, [self.min_x-x_margin, self.max_x+x_margin], [self.out_level+y_margin, self.in_level-y_margin])
        if opts.add_overview: self.update_range(self.overview_sp_col, [self.min_x-x_margin*2, self.max_x+x_margin*2], [self.out_level+y_margin*2, self.in_level-y_margin*2])

        # Add table if selected
        if opts.add_table:
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
        
        # Add dropdown menu for marker sizes and colors
        self.fig.update_layout(updatemenus=[self.build_styling_menu(pad_t=8)])

        # Add slider if selected
        if opts.add_slider:
            total_levels = self.out_level - self.in_level + y_margin*2
            num_slider_steps = int(total_levels / opts.num_levels_displayed * 3.5)
            self.fig.update_layout(sliders=[self.build_overview_slider(pad_t=48, x_margin=x_margin, y_margin=y_margin, num_levels_displayed=opts.num_levels_displayed, num_steps=num_slider_steps)])
            self.update_range(self.main_sp_col, [self.min_x-x_margin, self.max_x+x_margin], [self.in_level+opts.num_levels_displayed-y_margin, self.in_level-y_margin])

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
        menu_buttons = [self.build_menu_button(size_by, color_by) for (size_by, color_by) in self.size_color_options]
        active = self.size_color_idx
        return dict(buttons=menu_buttons, showactive=True, direction="up",
                    pad=dict(l=0, r=0, t=pad_t, b=0),
                    x=0, xanchor="left",
                    y=0, yanchor="top",
                    active=active)

    def build_overview_slider(self, pad_t=0, x_margin=0.5, y_margin=0.25, num_levels_displayed=2, num_steps=20):
        steps = []
        yaxis_varname = self.ax_var_name('yaxis', self.main_sp_col)
        for i in np.linspace(self.in_level-y_margin, self.out_level+y_margin-num_levels_displayed, num_steps):
            step = dict(
                method="relayout",
                args=[dict({
                    "shapes":[dict(type="rect", xref="x", yref="y",line=dict(color="#2A3F5F", width=1), fillcolor='rgba(42,63,96,0.2)',
                        x0=self.min_x-x_margin, y0=i, x1=self.max_x+x_margin, y1=i+num_levels_displayed,
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

    def get_node_color(self, node:MavNode, color_by='operation'):
        # Allow passing in color_by externally to generate marker colors
        # for Plotly custom dropdown menus
        if color_by == 'operation': 
            return node.op_color
        elif color_by == 'flops': 
            return self.flops_to_norm_val(node.flops)
        elif color_by == 'params': 
            return self.params_to_norm_val(node.params)
        else: 
            raise ValueError(f'Unknown color style: {color_by}')
        
    def get_connection_coords(self, c:MavConnection):
        if use_straight_connection(c, self.g):
            return [c.from_node.x, c.to_node.x], [c.from_node.y, c.to_node.y]
        else:
            # The curved lines display nicely at some levels of zoom, but look awkward
            # at others, especially when the vertical dimension is zoomed out.
            # The segmented lines are more consistent and use fewer points.
            offset = c.offset if c.offset is not None else 0.4
            return segmented_line_coords(c.from_node.x, c.from_node.y, c.to_node.y, offset)
