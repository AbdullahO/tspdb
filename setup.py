from setuptools import setup, find_packages

setup(name='tspdb',
      version='0.1',
      description='Predictive Functionalities in PostgreSQL',
      author='Abdullah Alomar',
      license='MIT',
      packages=['tspdb', 'tspdb.tests', 'tspdb.tests.testdata', 'tspdb.tests.testdata.tables', 'tspdb.src', 'tspdb.src.algorithms', 'tspdb.src.data', 'tspdb.src.database_module', 'tspdb.src.pindex', 'tspdb.src.prediction_models', 'tspdb.src.synthcontrol', 'tspdb.src.algorithms.pymf' ],
      install_requires=['numpy','h5py', 'pandas','sklearn','scipy'],
      zip_safe=False,
      include_package_data=True,
      package_data={'tspdb': ['tests/testdata/tables/*.csv']})
