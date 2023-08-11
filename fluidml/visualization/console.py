import logging
import os
import pydoc
import sys
import tempfile
from typing import TYPE_CHECKING

from fluidml.visualization.ascii import create_graph_in_ascii

if TYPE_CHECKING:
    import networkx as nx


logger = logging.getLogger(__name__)


class FluidPager:
    """Uses the pager installed on the system."""

    IDEAL_PAGER = "less --chop-long-lines --clear-screen --RAW-CONTROL-CHARS"

    def __init__(self):
        self.use_pager = None

        self._pager = self.get_pager()

    def get_pager(self):
        """Decide what method to use for paging through text."""
        if (
            not hasattr(sys.stdin, "isatty")
            or not hasattr(sys.stdout, "isatty")
            or not sys.stdin.isatty()
            or not sys.stdout.isatty()
        ):
            logger.warning("Console does not support paging. Defaulting to print graph.")
            return pydoc.plainpager

        self.use_pager = os.environ.get("MANPAGER") or os.environ.get("PAGER")
        if self.use_pager:
            if "less" in self.use_pager:
                self.use_pager = FluidPager.IDEAL_PAGER
            if sys.platform == "win32":  # pipes completely broken in Windows
                return lambda text: pydoc.tempfilepager(pydoc.plain(text), self.use_pager)
            elif os.environ.get("TERM") in ("dumb", "emacs"):
                return lambda text: pydoc.pipepager(pydoc.plain(text), self.use_pager)
            else:
                return lambda text: pydoc.pipepager(text, self.use_pager)
        if os.environ.get("TERM") in ("dumb", "emacs"):
            logger.warning("Console does not support paging. Defaulting to print graph.")
            return pydoc.plainpager
        if sys.platform == "win32":
            logger.warning(
                '"less" pager could not be found on Windows. Defaulting to "more". '
                'Install "less" for a better graph visualization experience.'
            )
            return lambda text: pydoc.tempfilepager(pydoc.plain(text), "more <")
        if hasattr(os, "system") and os.system("(less) 2>/dev/null") == 0:
            self.use_pager = FluidPager.IDEAL_PAGER
            return lambda text: pydoc.pipepager(text, self.use_pager)

        logger.warning(
            '"less" pager could not be found. Defaulting to "more" or, if vot available, "ttypager". '
            'Install "less" for a better graph visualization experience.'
        )
        (fd, filename) = tempfile.mkstemp()
        os.close(fd)
        try:
            if hasattr(os, "system") and os.system('more "%s"' % filename) == 0:
                return lambda text: pydoc.pipepager(text, "more")
            else:
                return pydoc.ttypager
        finally:
            os.unlink(filename)

    def show(self, content: str) -> None:
        """Use the same pager used by pydoc."""
        self._pager(content)


def visualize_graph_in_console(graph: "nx.DiGraph", use_pager: bool = True, use_unicode: bool = False):
    """Visualizes the task graph by rendering it to the console.

    When using a pager the keyboard input ":q" is required to continue.

    Args:
        graph: A networkx directed graph object
        use_pager: If true, tries rendering via pager (defaulting to print), if false, print
        use_unicode: Renders the graph in unicode if console supports it
    """

    console_graph = f"{graph.name}\n" if graph.name else "Graph"
    console_graph += f"{create_graph_in_ascii(graph=graph, use_unicode=use_unicode)}\n"

    if use_pager:
        pager = FluidPager()
        pager.show(content=console_graph)
    else:
        print(console_graph)
