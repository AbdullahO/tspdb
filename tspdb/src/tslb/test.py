from tslb import *


def import_data(string):
    train = pd.read_csv("data/tspdb_data/{}_train.csv".format(string))
    test = pd.read_csv("data/tspdb_data/{}_test.csv".format(string))
    return train, test

train, test = import_data('electricity')
print(train['y_train'])
print(run_test(test['y'].iloc[:], samples=300, k=2, discretization_method='change'))