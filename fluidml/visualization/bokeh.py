import networkx as nx

from bokeh.io import output_file, show
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


def visualize_graph_interactive(graph: nx.Graph, plot_width: int, plot_height: int,
                                node_width: int, node_height: int):
    # reformat the graph with attributes
    reformatted_graph = reformat_graph(graph)

    # get sugiyama layout
    layout = build_sugiyama_layout(reformatted_graph, 10, node_height, node_width)
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
    #plot.sizing_mode = "scale_both"

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

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
    plot.renderers.append(graph_renderer)

    #output_file("interactive_graphs.html")
    show(plot)