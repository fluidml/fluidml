import os
from importlib.util import module_from_spec, spec_from_file_location
from setuptools import setup, find_packages


_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(file_name: str, pkg="fluidml"):
    spec = spec_from_file_location(os.path.join(pkg, file_name), os.path.join(_PATH_ROOT, pkg, file_name))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

test_deps = ["pytest"]

setup(
    name="fluidml",
    version=about.__version__,
    description=about.__docs__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url=about.__homepage__,
    license=about.__license__,
    copyright=about.__copyright__,
    keywords=["pipelines", "machine-learning", "parallel", "deep-learning"],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "bokeh",  # enables interactive graph rendering in jupyter
        'dataclasses;python_version<"3.7"',  # backport for python versions without dataclasses
        "grandalf",  # for console graph visualization
        "metadict",  # enables attribute access for dicts (converted to MetaDict objects)
        "networkx",  # creation of task graph
        "rich",  # beautiful error traceback printing and logging
        "tblib",  # enables sending tracebacks through multiprocessing queue
    ],
    extras_require={
        "examples": [
            "datasets",
            "flair",
            "jupyterlab",
            "numpy",
            "pyyaml",
            "requests",
            "sklearn",
            "tokenizers>=0.10.1",
            "torchtext>=0.8.1",
            "torch",
            "tqdm",
        ],
        "mongo-store": ["mongoengine"],
        "tests": test_deps,
    },
    tests_require=test_deps,
)
