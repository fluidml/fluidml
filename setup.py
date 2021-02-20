from setuptools import setup, find_packages

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
builtins.__FLUIDML_SETUP__ = True

from fluidml import __version__, __author__, __author_email__, __license__, __copyright__, __homepage__, __docs__

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='fluidml',
      version=__version__,
      description=__docs__,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=__author__,
      author_email=__author_email__,
      url=__homepage__,
      download_url=__homepage__,
      license=__license__,
      copyright=__copyright__,
      keywords=['pipelines', 'machine-learning', 'parallel', 'deep-learning'],
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
          'dataclasses;python_version<"3.7"',  # backport for python versions without dataclasses
          'networkx',                          # creation of task graph
      ],
      extras_require={'examples': ['datasets',
                                   'flair',
                                   'jupyterlab',
                                   'numpy',
                                   'pyyaml',
                                   'requests',
                                   'sklearn',
                                   'tokenizers>=0.10.1',
                                   'torchtext>=0.8.1',
                                   'torch',
                                   'tqdm'],
                      'mongo-store': ['mongoengine'],
                      'rich-logging': ['rich',  # beautiful error traceback printing and logging
                                       'tblib', # enables sending tracebacks through multiprocessing queue
                                       ]},
      tests_require=['pytest'])
