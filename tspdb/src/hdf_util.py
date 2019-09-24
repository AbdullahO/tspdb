import h5py
import numpy as np
from math import ceil

def read_data(filename):
    return h5py.File(filename, "r+")


def write_data(filename, datalabel, matrix,mode = 'w'):
    f = h5py.File(filename, mode)
    f.create_dataset(datalabel, data=matrix)
    return True

def write_randomn_data(filename, matrixname, N, M, mean, sd):

    M = np.float64(np.random.normal(mean, sd, [N, M]))

    f = h5py.File(filename, "w")
    f.create_dataset(matrixname, data=M)

    f.close()
    return True

def write_randomn_data_seg(filename, matrixname, N, M, mean, sd, segment = None, max_memory = 10000*10000):
    if segment == None:
        segment = int(ceil(float(N)*M/max_memory))
    dm = M/segment
    m1 = int(M - (segment-1)*dm)
    M = np.float64(np.random.normal(mean, sd, [N,m1 ]))

    f = h5py.File(filename, "w")
    f.create_dataset(matrixname, data=M,  maxshape=(None,None))
    for i in range(1,segment):
        M = np.float64(np.random.normal(mean, sd, [N, dm]))
        f[matrixname].resize(m1 + i * dm, axis=1)
        f[matrixname][-M.shape[0]:,-M.shape[1]:] = M
    f.close()
    return True

def copy_data(SourceFileName,dataName, filenameCopy):

    f = h5py.File(filenameCopy, "w")
    SourceFileName.copy(dataName,f)
    f.close()

    return

def copy_data_legacy(A, filenameCopy, matrixnameCopy):
    (n, m) = np.shape(A)
    f = h5py.File(filenameCopy, "w")
    f.create_dataset(matrixnameCopy, data=A)
    f.close()
    return
def transpose_data(A, filename, matrixname):
    (n, m) = np.shape(A)
    f = h5py.File(filename, "w")
    f.create_dataset(matrixname, data=A[:].T)
    f.close()


