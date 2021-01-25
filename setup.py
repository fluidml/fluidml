from setuptools import setup, find_packages

setup(name='fluidml',
      version='0.0.1',
      description='A minimal framework for parallelizing machine learning tasks in python.',
      author='Rajkumar Ramamurthy, Lars Hillebrand',
      author_email='raj1514@gmail.com',
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires=['networkx', 'rich'],
      extras_require={'examples': ['datasets',
                                   'flair',
                                   'jupyterlab',
                                   'numpy',
                                   'pyyaml',
                                   'requests',
                                   'sklearn',
                                   'tokenizers',
                                   'torchtext',
                                   'torch',
                                   'tqdm'],
                      'mongo-store': ['mongoengine']},
      tests_require=['pytest'])
