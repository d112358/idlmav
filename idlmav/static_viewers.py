from .mavtypes import MavNode, MavGraph, MavConnection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import plotly.graph_objects as go
from .renderers.figure_renderer import use_straight_connection

class ArcViewer:
    """
    Simple viewer using matplotlib and arcs to connect nodes
    No interactive visualization
    """
    def __init__(self, g:MavGraph):
        self.g = g

    def draw(self):
        g = self.g
        fig, ax = plt.subplots()
        plt.close()
        x_lim, y_lim = (0,0), (0,0)
        for c in g.connections:
            xdata, ydata = self.get_connection_arc_coords(c.from_node, c.to_node)
            line = lines.Line2D(xdata=xdata, ydata=ydata, linewidth=1, color=[0,0,0])
            ax.add_line(line)
        for n in g.nodes:
            rect = patches.Rectangle((n.x - g.nw/2, n.y - g.nh/2), g.nw, g.nh, 
                                    linewidth=1, edgecolor=[0,0,0], facecolor=[1,1,1])
            ax.add_patch(rect)
            x_lim = (min((x_lim[0], n.x-1)), max((x_lim[1], n.x+1)))
            y_lim = (min((y_lim[0], n.y-1)), max((y_lim[1], n.y+1)))
            ax.text(n.x, n.y, n.name, horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(left=x_lim[0], right=x_lim[1])
        ax.set_ylim(bottom=y_lim[1], top=y_lim[0])
        ax.set_yticks([])
        ax.set_xticks([])
        return fig

    def append_arc_coords(self, x, y, cx, cy, r, quadrant, ccw, num_points):
        if ccw:
            t = np.linspace(np.pi/2*(quadrant-1), np.pi/2*quadrant, num_points)
        else:
            t = np.linspace(np.pi/2*quadrant, np.pi/2*(quadrant-1), num_points)
        xdata = cx + r*np.cos(t)
        ydata = cy - r*np.sin(t)  # Negative because y-axis is inverted on graph
        x += list(xdata)
        y += list(ydata)

    def get_connection_arc_coords(self, n0:MavNode, n1:MavNode):

        # Local parameters
        r = 0.1  # Arc radius
        na = 10  # Number of points per arc

        # Aliases
        g = self.g
        x0, y0, x1, y1 = n0.x, n0.y, n1.x, n1.y

        # Determine x-coordinate of main vertical segment
        nodes_in_x0_line = [n for n in g.nodes if n.x == x0 and n.y > y0 and n.y < y1]
        nodes_in_x1_line = [n for n in g.nodes if n.x == x1 and n.y > y0 and n.y < y1]
        if x0 == x1:
            # Vertical connection: bend very slightly if other blocks are in the way
            if nodes_in_x0_line:
                if x0 >= 0:
                    x_main = x0 + 0.5 - (1-g.nw)*0.4
                else:
                    x_main = x0 - 0.5 + (1-g.nw)*0.4
            else:
                x_main = x0
        elif abs(x0) <= abs(x1):
            # Moving from centre towards border
            if not nodes_in_x1_line:
                # Nothing in the way above output node
                x_main = x1
            elif not nodes_in_x0_line:
                # Nothing in the way below input node
                x_main = x0
            elif x0 < x1:
                # Moving from centre towards right border
                x_main = x1 - 0.5 + (1-g.nw)*0.1
            else:
                # Moving from centre towards left border
                x_main = x1 + 0.5 - (1-g.nw)*0.1
        else:
            # Moving from border in towards centre
            if not nodes_in_x0_line:
                # Nothing in the way below input node
                x_main = x0
            elif not nodes_in_x1_line:
                # Nothing in the way above output node
                x_main = x1
            elif x0 < x1:
                # Moving from left border towards centre
                x_main = x0 + 0.5 + (1-g.nw)*0.1
            else:
                # Moving from right border towards centre
                x_main = x0 - 0.5 - (1-g.nw)*0.1
                
        # Draw 4 arcs: 
        # * n0 to upper horizontal line
        # * upper horizontal line to main vertical line
        # * main vertical line to lower horizontal line
        # * lower horizontal line to n1
        # If coordinates for arcs are placed in the correct order, straight lines will automatically connect them
        xdata, ydata = [x0], [y0+g.nh/2]

        # n0 to upper horizontal line
        if x0 < x_main:
            # CCW 3rd quadrant arc with centre to the right of x0
            cx, cy, q, ccw = x0+r, y0+g.nh/2, 3, True
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)
        elif x0 > x_main:
            # Clockwise 4th quadrant arc with centre to the left of x0
            cx, cy, q, ccw = x0-r, y0+g.nh/2, 4, False
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)
        
        # Upper horizontal to main vertical line
        if x0 < x_main:
            # Clockwise 1st quadrant arc with centre to the left of x_main
            cx, cy, q, ccw = x_main-r, y0+g.nh/2+r+r, 1, False
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)
        elif x0 > x_main:
            # CCW 2nd quadrant arc with centre to the right of x_main
            cx, cy, q, ccw = x_main+r, y0+g.nh/2+r+r, 2, True
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)

        # main vertical line to lower horizontal line
        if x_main < x1:
            # CCW 3rd quadrant arc with centre to the right of x_main
            cx, cy, q, ccw = x_main+r, y1-g.nh/2-r-r, 3, True
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)
        elif x_main > x1:
            # Clockwise 4th quadrant arc with centre to the left of x_main
            cx, cy, q, ccw = x_main-r, y1-g.nh/2-r-r, 4, False
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)

        # lower horizontal line to n1
        if x_main < x1:
            # Clockwise 1st quadrant arc with centre to the left of x1
            cx, cy, q, ccw = x1-r, y1-g.nh/2, 1, False
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)
        elif x_main > x1:
            # CCW 2nd quadrant arc with centre to the right of x1
            cx, cy, q, ccw = x1+r, y1-g.nh/2, 2, True
            self.append_arc_coords(xdata, ydata, cx, cy, r, q, ccw, na)
        
        xdata += [x1]
        ydata += [y1-g.nh/2]
        return xdata,ydata

class PlotlyViewer:
    """
    Simple viewer using plotly
    Tooltips available on hover, but not other interaction
    """
    def __init__(self, g:MavGraph):
        self.g = g

    def draw(self):
        g = self.g
    
        # Draw nodes
        fig = go.Figure(
            data=go.Scatter(
                x=[n.x for n in g.nodes], 
                y=[n.y for n in g.nodes], 
                mode='markers', 
                marker=dict(
                    size=[1+3*np.log10(n.params) for n in g.nodes],
                    color=[np.log10(n.flops) for n in g.nodes],
                    colorscale='Bluered'
                ),
                hovertemplate=(
                    'Name: %{customdata[0]}<br>' +
                    'Activations: %{customdata[1]}<br>' +
                    'Parameters: %{customdata[2]}<br>' +
                    'FLOPS: %{customdata[3]}'
                ),
                customdata=[(n.name, n.activations, n.params, n.flops) for n in g.nodes],
                showlegend=False
            ),
            layout=go.Layout(
                xaxis=dict(title=None),
                yaxis=dict(title=None),
                title=None,
                width=500,
                height=500
            )
        )
        fig.update_xaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        fig.update_yaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])

        # Display direction
        in_level = g.in_nodes[0].y
        out_level = g.out_nodes[0].y
        fig.update_yaxes(range=[out_level+0.5, in_level-0.5])

        # Add connections
        for c in g.connections:
            fig.add_trace(
                go.Scatter(
                    x=[c.from_node.x, c.to_node.x],  # X coordinates of the two points in the pair
                    y=[c.from_node.y, c.to_node.y],  # Y coordinates of the two points in the pair
                    mode="lines",
                    line=dict(color="gray", width=1),
                    showlegend=False
                )
            )
        
        return fig


class SemiCircleViewer:
    """
    Draws all connections as semi-circles. Useful for visualizing
    skip connections between nodes in a straight line. 
    Tooltips available on hover, but not other interaction
    """
    def __init__(self, g:MavGraph):
        self.g = g

    def draw(self, fig_size=300):
        g = self.g
    
        # Draw nodes
        fig = go.Figure(
            data=go.Scatter(
                x=[n.x for n in g.nodes], 
                y=[n.y for n in g.nodes], 
                mode='markers', 
                marker=dict(
                    size=[1+3*np.log10(n.params) for n in g.nodes],
                    color=[np.log10(n.flops) for n in g.nodes],
                    colorscale='Bluered'
                ),
                hovertemplate=(
                    'Name: %{customdata[0]}<br>' +
                    'Activations: %{customdata[1]}<br>' +
                    'Parameters: %{customdata[2]}<br>' +
                    'FLOPS: %{customdata[3]}'
                ),
                customdata=[(n.name, n.activations, n.params, n.flops) for n in g.nodes],
                showlegend=False
            ),
            layout=go.Layout(
                xaxis=dict(title=None),
                yaxis=dict(title=None),
                title=None,
                width=fig_size,
                height=fig_size,
            )
        )
        fig.update_xaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])
        fig.update_yaxes(showgrid=False, zeroline=False, tickmode='array', tickvals=[])

        # Display direction
        in_level = min([n.y for n in g.nodes])
        out_level = max([n.y for n in g.nodes])
        fig.update_yaxes(range=[out_level+0.5, in_level-0.5])

        # Add connections
        for ci,c in enumerate(g.connections):
            xs, ys = self.get_connection_coords(c)

            line_trace = go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="gray", width=1),            
                hoverinfo='skip',
                hovertemplate=
                    f'Connection {ci}<br>' +
                    f'Nodes: {c.from_node.name}-{c.to_node.name}<br>' +
                    '<extra></extra>',
                showlegend=False
            )
            fig.add_trace(line_trace)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_layout(modebar=dict(remove=["toimage", "select", "lasso", "zoomin", "zoomout"], orientation="v"))
        
        return fig
    
    def get_connection_coords(self, c:MavConnection):
        cx = (c.from_node.x + c.to_node.x)/2 
        cy = (c.from_node.y + c.to_node.y)/2   
        if use_straight_connection(c, self.g):
            return [c.from_node.x, cx, c.to_node.x], [c.from_node.y, cy, c.to_node.y]
        num_points = 20  # Number of points per arc
        r = (c.to_node.y - c.from_node.y)/2  # Arc radius
        t = np.linspace(-np.pi/2, np.pi/2, num_points)
  
        if c.offset is not None and c.offset < 0:
            xdata = cx - r*np.cos(t)
            ydata = cy - r*np.sin(t)  # Negative because y-axis is inverted on graph
        else:
            xdata = cx + r*np.cos(t)
            ydata = cy - r*np.sin(t)  # Negative because y-axis is inverted on graph
        return list(xdata), list(ydata)