import logging
import os

from .__about__ import *
from .flow import Flow, TaskSpec
from .common import Task, configure_logging


logging.getLogger(__name__)

package_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(package_path)
