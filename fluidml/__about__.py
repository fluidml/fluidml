import time

_this_year = time.strftime("%Y")
__version__ = "0.3.4"
__author__ = "Lars Hillebrand, Rajkumar Ramamurthy"
__author_email__ = "hokage555@web.de"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-{_this_year}, {__author__}."
__homepage__ = "https://github.com/fluidml/fluidml/"
__docs__ = (
    "FluidML is a lightweight framework for developing machine learning pipelines."
    " Focus only on your tasks and not the boilerplate!"
)

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__homepage__",
    "__license__",
    "__version__",
]
