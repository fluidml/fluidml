from typing import Optional

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

    HEIGHT = 3

    def __init__(self, name: str, height: Optional[int] = None, width: Optional[int] = None):
        # height of the node (top and bottom box edges + name).
        self.h = height if height is not None else self.HEIGHT
        # width of the node (right and left bottom edges + name).
        self.w = width if width is not None else len(name) + 2


def _build_sugiyama_layout(
    graph: DiGraph,
    iterations: int = 1,
    node_height: Optional[int] = None,
    node_width: Optional[int] = None,
):
    """Function to build a sugiyama layout for a graph

    Just a reminder about naming conventions:
    +------------X
    |
    |
    |
    |
    Y
    """

    vertexes = {v: Vertex(f" {v} ") for v in list(graph.nodes())}
    edges = [Edge(vertexes[s], vertexes[e]) for s, e in list(graph.edges())]
    vertexes = vertexes.values()

    graph = Graph(vertexes, edges)

    for vertex in vertexes:
        vertex.view = VertexViewer(vertex.data, node_height, node_width)

    for edge in edges:
        edge.view = EdgeViewer()

    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))

    sug.init_all(roots=roots, optimize=True)

    # vertical space between nodes
    max_num_layer_nodes = max([len(layer) for layer in sug.layers])
    minh = max(max_num_layer_nodes, node_height if node_height else VertexViewer.HEIGHT)
    sug.yspace = minh

    # horizontal space between nodes
    # determine min box length to create the best layout
    minw = min(v.view.w for v in vertexes)
    sug.xspace = minw
    sug.route_edge = route_with_lines

    sug.draw(iterations)

    return sug
