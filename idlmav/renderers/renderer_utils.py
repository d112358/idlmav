from ..mavtypes import MavGraph, MavConnection
import numpy as np

def use_straight_connection(c:MavConnection, g:MavGraph):
    """
    `use_straight_connection` returns True if the nodes in `c` 
    should be connected by a straight line and False otherwise.
    It will return False if `g` has another node exactly in
    the path of a straight line.
    """
    # TODO: Update the following functions when supporting horizontal layouts, UNets, etc.
    # * use_straight_connection
    # * segmented_line_coords
    # * curved_line_coords
    # * append_arc_coords
    # The current implementations only consider vertical graph layout
    n0, n1 = c.from_node, c.to_node
    x0, y0, x1, y1 = n0.x, n0.y, n1.x, n1.y
    if x0 != x1: return True  # Use straght lines unless vertical and obstructed by another node
    nodes_on_line = [n for n in g.nodes if n.x == x0]  # First just perform one check on all nodes
    nodes_on_segment = [n for n in nodes_on_line if n.y > y0 and n.y < y1]  # Perform other 2 checks on subset of nodes
    return False if nodes_on_segment else True
        
def segmented_line_coords(x01, y0, y1, offset=0.4):
    """
    `segmented_line_coords` returns the x and y coordinates for a
    segmented line that connects two nodes on the same horizontal
    coordinate `x01` and different vertical coordinates `y0` and 
    `y1`. This is an alternative to `curved_line_coords`.
    """
    ymin, ymax = min([y0,y1]), max([y0,y1])
    dy = min((ymax-ymin)*0.3, 0.5)
    x = [x01,  x01+offset,  x01+offset,  x01]
    y = [ymin, ymin+dy, ymax-dy, ymax]
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
