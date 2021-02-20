import logging
import time

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    _ = None if __FLUIDML_SETUP__ else None
except NameError:
    __FLUIDML_SETUP__: bool = False

if not __FLUIDML_SETUP__:
    from .flow import Flow
    from .swarm import Swarm

_this_year = time.strftime("%Y")
__version__ = '0.1.3'
__author__ = 'Rajkumar Ramamurthy, Lars Hillebrand'
__author_email__ = 'raj1514@gmail.com'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2020-{_this_year}, {__author__}.'
__homepage__ = 'https://github.com/fluidml/fluidml/'
__docs__ = (
    "FluidML is a lightweight framework for developing machine learning pipelines."
    " Focus only on your tasks and not the boilerplate!"
)

logging.getLogger(__name__)
