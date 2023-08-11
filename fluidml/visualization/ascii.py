"""Draws DAG in ASCII.
Note: This script is largely taken from the DVC project.
See: https://github.com/iterative/dvc/blob/master/dvc/dagascii.py
"""


import logging
import math
import os
from typing import TYPE_CHECKING, Dict

from fluidml.visualization.graph_layout import _build_sugiyama_layout

if TYPE_CHECKING:
    import networkx as nx
    from grandalf.layouts import SugiyamaLayout


logger = logging.getLogger(__name__)


class AsciiCanvas:
    """Class for drawing graph in ASCII."""

    TIMEOUT = 10

    def __init__(
        self,
        sug: "SugiyamaLayout",
        cols: int,
        rows: int,
        chars: Dict[str, str],
        minx: int,
        miny: int,
    ):
        """Initializes class for drawing graph in ASCII.

        Args:
            sug: Calculated sugiyama layout for the input graph.
            cols: Number of columns in the canvas. Should be > 1.
            rows: Number of rows in the canvas. Should be > 1.
            chars: Character set used for graph rendering (can be ascii or unicode).
            minx: Minimum x coordinate drawn on canvas.
            miny: Minimum y coordinate drawn on canvas.
        """

        assert cols > 1
        assert rows > 1

        self.sug = sug
        self.cols = cols
        self.rows = rows
        self.chars = chars
        self.minx = minx
        self.miny = miny

        # create an empty canvas num_lines x num_cols
        self.canvas = [[" "] * cols for _ in range(rows)]

    def point(self, x: int, y: int, char: str):
        """Create a point on ASCII canvas.
        Args:
            x: x coordinate. Should be >= 0 and < number of columns in the canvas.
            y: y coordinate. Should be >= 0 an < number of lines in the canvas.
            char: character to place in the specified point on the canvas.
        """
        assert len(char) == 1
        assert x >= 0
        assert x < self.cols
        assert y >= 0
        assert y < self.rows

        self.canvas[y][x] = char

    def line(self, x0: int, y0: int, x1: int, y1: int, char: str):
        """Create a line on ASCII canvas.
        Args:
            x0: x coordinate where the line should start.
            y0: y coordinate where the line should start.
            x1: x coordinate where the line should end.
            y1: y coordinate where the line should end.
            char: character to draw the line with.
        """

        if x0 > x1:
            x1, x0 = x0, x1
            y1, y0 = y0, y1

        dx = x1 - x0
        dy = y1 - y0

        if dx == 0 and dy == 0:
            self.point(x0, y0, char)
        elif abs(dx) >= abs(dy):
            for x in range(x0, x1 + 1):
                if dx == 0:
                    y = y0
                else:
                    y = y0 + int(round((x - x0) * dy / float(dx)))
                self.point(x, y, char)
        elif y0 < y1:
            for y in range(y0, y1 + 1):
                if dy == 0:
                    x = x0
                else:
                    x = x0 + int(round((y - y0) * dx / float(dy)))
                self.point(x, y, char)
        else:
            for y in range(y1, y0 + 1):
                if dy == 0:
                    x = x0
                else:
                    x = x1 + int(round((y - y1) * dx / float(dy)))
                self.point(x, y, char)

    def text(self, x: int, y: int, text: str):
        """Print a text on ASCII canvas.
        Args:
            x: x coordinate where the text should start.
            y: y coordinate where the text should start.
            text: string that should be printed.
        """
        for i, char in enumerate(text):
            self.point(x + i, y, char)

    def box(self, x0: int, y0: int, width: int, height: int):
        """Create a box on ASCII canvas.
        Args:
            x0: x coordinate of the box corner.
            y0: y coordinate of the box corner.
            width: box width.
            height: box height.
        """
        assert width > 1
        assert height > 1

        width -= 1
        height -= 1

        for x in range(x0, x0 + width):
            self.point(x, y0, self.chars["horizontal_box"])
            self.point(x, y0 + height, self.chars["horizontal_box"])

        for y in range(y0, y0 + height):
            self.point(x0, y, self.chars["vertical_box"])
            self.point(x0 + width, y, self.chars["vertical_box"])

        self.point(x0, y0, self.chars["top_left_box"])
        self.point(x0 + width, y0, self.chars["top_right_box"])
        self.point(x0, y0 + height, self.chars["bottom_left_box"])
        self.point(x0 + width, y0 + height, self.chars["bottom_right_box"])

    def draw_edges(self):
        for edge in self.sug.g.sE:
            assert len(edge.view._pts) > 1
            for index in range(1, len(edge.view._pts)):
                start = edge.view._pts[index - 1]
                end = edge.view._pts[index]

                start_x = int(round(start[0] - self.minx))
                start_y = int(round(start[1] - self.miny))
                end_x = int(round(end[0] - self.minx))
                end_y = int(round(end[1] - self.miny))

                assert start_x >= 0
                assert start_y >= 0
                assert end_x >= 0
                assert end_y >= 0

                self.line(start_x, start_y, end_x, end_y, self.chars["line"])

    def draw_nodes(self):
        for vertex in self.sug.g.sV:
            # NOTE: moving boxes w/2 to the left
            x = vertex.view.xy[0] - vertex.view.w / 2.0
            y = vertex.view.xy[1]

            self.box(
                int(round(x - self.minx)),
                int(round(y - self.miny)),
                vertex.view.w,
                vertex.view.h,
            )

            self.text(
                int(round(x - self.minx)) + 1,
                int(round(y - self.miny)) + 1,
                vertex.data,
            )

    def to_str(self) -> str:
        """Draws ASCII canvas on the screen."""
        lines = map("".join, self.canvas)
        joined_lines = os.linesep.join(lines)
        return joined_lines


def _get_graph_char_set(use_unicode: bool) -> Dict[str, str]:
    chars = {
        "line": "*",
        "top_left_box": "+",
        "top_right_box": "+",
        "bottom_left_box": "+",
        "bottom_right_box": "+",
        "horizontal_box": "-",
        "vertical_box": "|",
    }

    if use_unicode:
        try:
            import sys

            "╭╮╰╯·─|".encode(sys.stdout.encoding).decode(sys.stdout.encoding)
            chars = {
                "line": "·",
                "top_left_box": "╭",
                "top_right_box": "╮",
                "bottom_left_box": "╰",
                "bottom_right_box": "╯",
                "horizontal_box": "─",
                "vertical_box": "│",
            }

        except UnicodeEncodeError:
            logger.warning(f"Console does not support unicode chars. Defaulting to ascii.")
    return chars


def create_graph_in_ascii(graph: "nx.DiGraph", use_unicode: bool = False) -> str:
    """Create ascii (or unicode) graph and return as str ready for printing.
    Args:
        graph (DiGraph): a networkx directed graph object
        use_unicode (bool): renders the graph in unicode if console supports it
    """

    chars = _get_graph_char_set(use_unicode=use_unicode)

    sug = _build_sugiyama_layout(graph=graph)

    # NOTE: coordinates might be negative, so we need to shift
    # everything to the positive plane before we actually draw the graph.
    xs, ys = [], []
    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        xs.append(vertex.view.xy[0] - vertex.view.w / 2.0)
        xs.append(vertex.view.xy[0] + vertex.view.w / 2.0)
        ys.append(vertex.view.xy[1])
        ys.append(vertex.view.xy[1] + vertex.view.h)

    for edge in sug.g.sE:
        for x, y in edge.view._pts:
            xs.append(x)
            ys.append(y)

    minx = min(xs)
    miny = min(ys)
    maxx = max(xs)
    maxy = max(ys)

    canvas_cols = int(math.ceil(math.ceil(maxx) - math.floor(minx))) + 1
    canvas_lines = int(round(maxy - miny))

    canvas = AsciiCanvas(sug, canvas_cols, canvas_lines, chars, minx, miny)

    # first draw edges so that node boxes could overwrite them
    canvas.draw_edges()
    canvas.draw_nodes()

    return canvas.to_str()
