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


def get_edges(sug_layout, height):
    xs, ys = [], []
    for edge in sug_layout.g.sE:
        for x, y in edge.view._pts:
            xs.append(x)
            ys.append(height - y - 1)
    minx = min(xs)
    miny = min(ys)

    xs, ys = [], []
    for edge in sug_layout.g.sE:
        assert len(edge.view._pts) > 1
        node_x, node_y = [], []
        for index in range(1, len(edge.view._pts)):
            start = edge.view._pts[index - 1]
            end = edge.view._pts[index]
            node_x.append(start[0])
            node_x.append(end[0])
            node_y.append(height - start[1] - 1)
            node_y.append(height - end[1] - 1)

        xs.append(node_x)
        ys.append(node_y)
    return xs, ys


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

    # # edges from the layout
    xs, ys = get_edges(layout, plot_height)
    graph_renderer.edge_renderer.data_source.data['xs'] = xs
    graph_renderer.edge_renderer.data_source.data['ys'] = ys

    #graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
    plot.renderers.append(graph_renderer)

    #output_file("interactive_graphs.html")
    show(plot)