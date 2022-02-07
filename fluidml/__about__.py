import time

_this_year = time.strftime("%Y")
__version__ = '0.2.0'
__author__ = 'Rajkumar Ramamurthy, Lars Hillebrand'
__author_email__ = 'raj1514@gmail.com'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2020-{_this_year}, {__author__}.'
__homepage__ = 'https://github.com/fluidml/fluidml/'
__docs__ = (
    "FluidML is a lightweight framework for developing machine learning pipelines."
    " Focus only on your tasks and not the boilerplate!"
)

__all__ = ["__author__", "__author_email__", "__copyright__", "__docs__", "__homepage__", "__license__", "__version__"]
