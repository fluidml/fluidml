import networkx as nx

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool, BoxSelectTool, PanTool, WheelZoomTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx
from bokeh.models import ColumnDataSource, LabelSet

from grandalf.graphs import Edge, Graph, Vertex
from grandalf.layouts import SugiyamaLayout
from grandalf.routing import EdgeViewer, route_with_lines
from networkx import DiGraph


class VertexViewer:
    """Class to define vertex box boundaries that will be accounted for during
    graph building by grandalf.
    Args:
        name (str): name of the vertex.
    """
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width


def build_sugiyama_layout(graph: DiGraph, node_height: int, node_width: int):
    """ Function to build a sugiyama layout for a graph
    """

    vertexes = {v: Vertex(f' {v} ') for v in list(graph.nodes())}
    edges = [Edge(vertexes[s], vertexes[e]) for s, e in list(graph.edges())]
    vertexes = vertexes.values()

    graph = Graph(vertexes, edges)

    for vertex in vertexes:
        vertex.view = VertexViewer(node_height, node_width)

    for edge in edges:
        edge.view = EdgeViewer()

    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))

    sug.init_all(roots=roots, optimize=True)

    # vertical space between nodes
    max_num_layer_nodes = max([len(layer) for layer in sug.layers])
    minh = max(max_num_layer_nodes, node_height)
    sug.yspace = minh

    # horizontal space between nodes
    # determine min box length to create the best layout
    minw = min(v.view.w for v in vertexes)
    sug.xspace = minw
    sug.route_edge = route_with_lines

    sug.draw(10)

    return sug


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


def visualize_graph_interactive(graph: nx.Graph, plot_width: int, plot_height: int,
                                node_width: int, node_height: int):
    # reformat the graph with attributes
    reformatted_graph = reformat_graph(graph)

    # get sugiyama layout
    layout = build_sugiyama_layout(reformatted_graph, node_height, node_width)
    positions = {vertex.data.strip(): (vertex.view.xy[0], vertex.view.xy[1]) for vertex in layout.g.sV}
    positions = flip_positions(positions, plot_height)

    # edge attributes
    edge_attrs = {}
    for start_node, end_node, _ in reformatted_graph.edges(data=True):
        edge_attrs[(start_node, end_node)] = "black"

    nx.set_edge_attributes(reformatted_graph, edge_attrs, "edge_color")

    # Show with Bokeh
    plot = Plot(plot_width=plot_width, plot_height=plot_height)
    plot.title.text = "Task Graph"
    plot.sizing_mode = "scale_both"

    node_hover_tool = HoverTool(tooltips=[("task_name", "@task_name")])
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), BoxSelectTool(), PanTool(), WheelZoomTool())

    graph_renderer = from_networkx(reformatted_graph, positions, center=(0, 0))

    x, y = zip(*positions.values())
    node_labels = nx.get_node_attributes(reformatted_graph, 'task_name')
    source = ColumnDataSource({'x': x, 'y': y,
                             'task_name': list(positions.keys())})
    labels = LabelSet(x='x', y='y', text='task_name', source=source, text_align="center",
                      background_fill_color='white', text_font_size="12px", border_line_color="black")
    plot.renderers.append(labels)

    #graph_renderer.node_renderer.glyph = Square(fill_color=Spectral4[0])
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
    plot.renderers.append(graph_renderer)

    #output_file("interactive_graphs.html")
    show(plot)