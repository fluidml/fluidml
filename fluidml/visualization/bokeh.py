import networkx as nx

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import (BoxZoomTool, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool, BoxSelectTool, PanTool, WheelZoomTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx
from bokeh.models import ColumnDataSource, LabelSet

from fluidml.visualization import build_sugiyama_layout


def reformat_graph(graph):
    reformatted_graph = nx.Graph()

    # add nodes
    for node in graph.nodes:
        reformatted_graph.add_node(node, task_name=node)

    # add edges
    for edge in graph.edges:
        reformatted_graph.add_edge(edge[0], edge[1])
    return reformatted_graph


def flip_positions(positons, height):
    flipped = {}
    for key, (x, y) in positons.items():
        flipped[key] = (x, height - y - 1)
    return flipped


def get_edges(sug_layout, height):
    xs, ys = [], []
    for edge in sug_layout.g.sE:
        assert len(edge.view._pts) > 1

        # get corresponding start and end nodes
        start_node, end_node = edge.v[0].view.xy, edge.v[1].view.xy
        
        # get computed multi-line edges
        node_x, node_y = [], []
        for index in range(1, len(edge.view._pts)):
            start = edge.view._pts[index - 1]
            end = edge.view._pts[index]
            node_x.append(start[0])
            node_x.append(end[0])
            node_y.append(height - start[1] - 1)
            node_y.append(height - end[1] - 1)
            #node_y.append(start[1])
            #node_y.append(end[1])
        
        # in these, we first have to manipulate the first and last co-ordindates
        # so that they have extend upto node positions
        node_x[0], node_y[0] = start_node[0], (height - start_node[1] - 1)
        node_x[-1], node_y[-1] = end_node[0], (height - end_node[1] - 1)

        xs.append(node_x)
        ys.append(node_y)
    return xs, ys


def visualize_graph_interactive(graph: nx.Graph, plot_width: int = 500, plot_height: int = 500,
                                node_width: int = 50, node_height: int = 50,
                                scale_width: bool = True):
    # reformat the graph with attributes
    reformatted_graph = reformat_graph(graph)

    # bokeh plot settings
    plot = figure(plot_width=plot_width, plot_height=plot_height)
    plot.title.text = "Task Graph"
    plot.grid.visible = False
    plot.sizing_mode = "scale_width" if scale_width else "auto"
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    # get sugiyama layout
    layout = build_sugiyama_layout(reformatted_graph, 10, node_height, node_width)
    positions = {vertex.data.strip(): (vertex.view.xy[0], vertex.view.xy[1]) for vertex in layout.g.sV}
    positions = flip_positions(positions, plot_height)

    # plot nodes
    x, y = zip(*positions.values())
    node_labels = nx.get_node_attributes(reformatted_graph, 'task_name')
    source = ColumnDataSource({'x': x, 'y': y,
                             'task_name': list(positions.keys())})
    labels = LabelSet(x='x', y='y', text='task_name', source=source, text_align="center",
                      background_fill_color='white', text_font_size="12px", border_line_color="black", name="task_name")
    plot.renderers.append(labels)

    # plot edges
    xs, ys = get_edges(layout, plot_height)
    plot.multi_line(xs, ys)

    show(plot)