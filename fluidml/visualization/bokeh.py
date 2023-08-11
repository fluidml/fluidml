from typing import Optional

import networkx as nx
from bokeh.io import show
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure

from fluidml.visualization.graph_layout import _build_sugiyama_layout


def reformat_graph(graph):
    reformatted_graph = nx.Graph()

    # add nodes
    for node in graph.nodes:
        reformatted_graph.add_node(node, task_name=node)

    # add edges
    for edge in graph.edges:
        reformatted_graph.add_edge(edge[0], edge[1])
    return reformatted_graph


def flip_positions(positions, height):
    flipped = {}
    for key, (x, y) in positions.items():
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

        # in these, we first have to manipulate the first and last co-ordindates
        # so that they have extend upto node positions
        node_x[0], node_y[0] = start_node[0], (height - start_node[1] - 1)
        node_x[-1], node_y[-1] = end_node[0], (height - end_node[1] - 1)

        xs.append(node_x)
        ys.append(node_y)
    return xs, ys


def visualize_graph_interactive(
    graph: nx.Graph,
    plot_width: int = 500,
    plot_height: int = 200,
    node_width: int = 50,
    node_height: int = 50,
    scale_width: bool = True,
    browser: Optional[str] = None,
):
    """Visualizes the task graph interactively in a browser or jupyter notebook.

    Args:
        graph: A networkx directed graph object
        plot_width: The width of the plot.
        plot_height: The height of the plot.
        node_width: Influences the horizontal space between nodes.
        node_height: Influences the vertical space between nodes.
        scale_width: If true, scales the graph to the screen width.
        browser: If provided, renders the graph in the browser, e.g. "chrome" or "firefox". Note the browser might need
            to be registered using Python's ``webbrowser`` library.
    """
    # reformat the graph with attributes
    reformatted_graph = reformat_graph(graph)

    # bokeh plot settings
    plot = figure(width=plot_width, height=plot_height)
    plot.title.text = "Task Graph"
    plot.title.text_font_size = "20pt"
    plot.grid.visible = False
    plot.sizing_mode = "scale_width" if scale_width else "auto"
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    # get sugiyama layout
    layout = _build_sugiyama_layout(reformatted_graph, 10, node_height, node_width)
    positions = {vertex.data.strip(): (vertex.view.xy[0], vertex.view.xy[1]) for vertex in layout.g.sV}
    positions = flip_positions(positions, plot_height)

    # get positions
    x, y = zip(*positions.values())

    # plot edges
    xs, ys = get_edges(layout, plot_height)
    plot.multi_line(xs, ys)

    # plot nodes
    source = ColumnDataSource({"x": x, "y": y, "task_name": list(positions.keys())})
    labels = LabelSet(
        x="x",
        y="y",
        text="task_name",
        source=source,
        text_align="center",
        background_fill_color="white",
        text_font_size="16px",
        border_line_color="black",
        name="task_name",
    )
    plot.renderers.append(labels)

    show(plot, browser=browser)
