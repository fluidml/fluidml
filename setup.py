from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='fluidml',
      version='0.1.0',
      description='A minimal framework for parallelizing machine learning tasks in python.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Rajkumar Ramamurthy, Lars Hillebrand',
      author_email='raj1514@gmail.com',
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires=['networkx'],
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
                      'mongo-store': ['mongoengine'],
                      'rich-logging': ['rich']},
      tests_require=['pytest'])
