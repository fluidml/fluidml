import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(file_name: str, pkg="fluidml"):
    spec = spec_from_file_location(os.path.join(pkg, file_name), os.path.join(_PATH_ROOT, pkg, file_name))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

test_deps = ["pytest>=7.0.0", "pytest-cov", "black>=22.6", "pre-commit>=2.17.0"]

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
    python_requires=">=3.7.0",
    package_data={"": ["*.txt"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: IPython",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Typing :: Typed",
    ],
    install_requires=[
        "bokeh>=2.4.3",  # enables interactive graph rendering in jupyter
        "grandalf>=0.7",  # for console graph visualization
        "metadict>=0.1.3",  # enables attribute access for dicts (converted to MetaDict objects)
        "networkx>=2.5",  # creation of task graph
        "pydantic>=2.0",  # for easy data model serialization and loading
        "rich>=9.13.0",  # beautiful error traceback printing and logging
        "tblib>=1.7.0",  # enables sending tracebacks through multiprocessing queue
    ],
    extras_require={
        "examples": [
            "datasets",
            "flair",
            "jupyterlab",
            "numpy",
            "pyyaml",
            "requests",
            "scikit-learn",
            "tokenizers>=0.10.1",
            "torchtext>=0.8.1",
            "torch",
            "tqdm",
        ],
        "mongo-store": ["mongoengine>=0.27.0"],
        "tests": test_deps,
    },
    tests_require=test_deps,
)
