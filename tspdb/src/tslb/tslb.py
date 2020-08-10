
from tspdb.src.tslb.src.continuous import *
from tspdb.src.tslb.src.lzw import *
from tspdb.src.tslb.src.regModel import regModel as regModel
from tspdb.src.tslb.src.utils import *


def get_diff(series):
    return (series.shift(-1) - series)[:-1]


def transform_data(series):

    y_t = (series.shift(-1) - series)[:-1]
    x_t = pd.Series(index = y_t[1:].index, dtype=int)
    x_t[y_t == 0] = 0
    x_t[y_t > 0] = 1
    x_t[y_t < 0] = 2
    return x_t

def get_lower_bound(test, samples=1000, k=3, discretization_method='quantization'):
    # discretize the sequence
    if discretization_method == 'quantization':
        n = k
        seq = get_diff(test)
        size = len(seq)
        discretized_seq, categories = discretize(seq, n)
    elif discretization_method == 'change':
        n = 3
        # transform
        size = len(test)
        discretized_seq  = transform_data(test)
    elif discretization_method == 'None':
        n = len(np.unique(test.values))
        # transform
        size = len(test)
        if n > 0.5*size:
            raise Exception ('The time series is not discrete, or there are not enough observations (no. observations < (alphabet size)/2 )')   
        discretized_seq = test
    else:
        raise ValueError ('Choose discretization_method from {"change", "quantization", "None"}')
    
    myRegModel = regModel(n, size, samples)
    myRegModel.fit()

    # convert format and get p_tilda
    uncomp_numbers = list(discretized_seq)
    ratio = lzw_compression_ratio(uncomp_numbers, n)
    ent = myRegModel.get_entropy(ratio)
    lb = h_inverse(ent, n, a=0.001)
    return lb
