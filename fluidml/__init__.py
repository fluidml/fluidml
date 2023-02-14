import logging
import os

from .__about__ import *
from .flow import Flow
from .swarm import Swarm
from .common import Task


logging.getLogger(__name__)

package_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(package_path)
