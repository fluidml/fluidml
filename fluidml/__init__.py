import logging
import time

from .flow import Flow
from .swarm import Swarm


logging.getLogger(__name__).addHandler(logging.NullHandler())


_this_year = time.strftime("%Y")
__version__ = '0.1.0'
__author__ = 'Rajkumar Ramamurthy, Lars Hillebrand'
__author_email__ = 'raj1514@gmail.com'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2020-{_this_year}, {__author__}.'
__homepage__ = 'https://github.com/fluidml/fluidml/'
__docs__ = (
    "FluidML is a lightweight framework for developing machine learning pipelines."
    " Focus only on your tasks and not the boilerplate!"
)
