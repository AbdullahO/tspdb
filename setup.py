from setuptools import setup, find_packages

setup(name='tspdb',
      version='0.2',
      description='Predictive Functionalities in PostgreSQL',
      author='Abdullah Alomar',
      license='Apache 2.0',
      packages=['tspdb', 'tspdb.tests', 'tspdb.tests.testdata', 'tspdb.tests.testdata.tables', 'tspdb.src', 'tspdb.src.algorithms', 'tspdb.src.data', 'tspdb.src.database_module', 'tspdb.src.pindex', 'tspdb.src.prediction_models', 'tspdb.src.synthcontrol', 'tspdb.src.algorithms.pymf', 'tspdb.src.tslb','tspdb.src.tslb.src' ],
      install_requires=['numpy','h5py', 'pandas','sklearn','scikit-learn', 'scipy','sqlalchemy'],
      zip_safe=False,
      include_package_data=True,
      package_data={'tspdb': ['tests/testdata/tables/*.csv']})
