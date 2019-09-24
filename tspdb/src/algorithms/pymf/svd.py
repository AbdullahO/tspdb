# Authors: Christian Thurau
# License: BSD 3 Clause
"""  
PyMF Singular Value Decomposition.

    SVD : Class for Singular Value Decomposition
    pinv() : Compute the pseudoinverse of a Matrix
     
"""
from numpy.linalg import eigh
import time
import scipy.sparse
import numpy as np

from base import PyMFBase3, eighk

try:
    import scipy.sparse.linalg.eigen.arpack as linalg
except (ImportError, AttributeError):
    import scipy.sparse.linalg as linalg


def pinv(A, k=-1, eps= np.finfo(float).eps):    
    # Compute Pseudoinverse of a matrix   
    svd_mdl =  SVD(A, k=k)
    svd_mdl.factorize()
    
    S = svd_mdl.S
    Sdiag = S.diagonal()
    Sdiag = np.where(Sdiag>eps, 1.0/Sdiag, 0.0)
    
    for i in range(S.shape[0]):
        S[i,i] = Sdiag[i]

    if scipy.sparse.issparse(A):            
        A_p = svd_mdl.V.transpose() * (S * svd_mdl.U.transpose())
    else:    
        A_p = np.dot(svd_mdl.V.T, np.core.multiply(np.diag(S)[:,np.newaxis], svd_mdl.U.T))

    return A_p


class SVD(PyMFBase3):    
    """      
    SVD(data, show_progress=False)
    
    
    Singular Value Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. U and V correspond to eigenvectors of the matrices
    data*data.T and data.T*data.
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV                
    
    Example
    -------
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> svd_mdl = SVD(data)    
    >>> svd_mdl.factorize()
    """

    def _compute_S(self, values):
        """
        """
        self.S = np.diag(np.sqrt(values))
        
        # and the inverse of it
        S_inv = np.diag(np.sqrt(values)**-1.0)
        return S_inv

   
    def factorize(self):    

        def _right_svd():            
            AA = np.dot(self.data[:,:], self.data[:,:].T)
                   # argsort sorts in ascending order -> access is backwards
            values, self.U = eighk(AA, k=self._k)

            # compute S
            self.S = np.diag(np.sqrt(values))
            
            # and the inverse of it
            S_inv = self._compute_S(values)
                    
            # compute V from it
            self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))
            
        
        def _left_svd():
            AA = np.dot(self.data[:,:].T, self.data[:,:])
            
            values, Vtmp = eighk(AA, k=self._k)

            # compute S
            self.S = np.diag(np.sqrt(values))

            self.V = Vtmp.T

            # and the inverse of it
            S_inv = self._compute_S(values)

            self.U = np.dot(np.dot(self.data[:,:], self.V.T), S_inv)
    
        def _sparse_right_svd():
            ## for some reasons arpack does not allow computation of rank(A) eigenvectors (??)    #
            AA = self.data*self.data.transpose()
            
            if self.data.shape[0] > 1:                    
                # only compute a few eigenvectors ...
                if self._k > 0 and self._k < self.data.shape[0]-1:
                    k = self._k
                else:
                    k = self.data.shape[0]-1
                values, u_vectors = linalg.eigsh(AA,k=k)
            else:                
                values, u_vectors = eigh(AA.todense())
            
            # get rid of negative/too low eigenvalues   
            s = np.where(values > self._EPS)[0]
            u_vectors = u_vectors[:, s] 
            values = values[s]
            
            # sort eigenvectors according to largest value
            # argsort sorts in ascending order -> access is backwards
            idx = np.argsort(values)[::-1]
            values = values[idx]                        
            
            self.U = scipy.sparse.csc_matrix(u_vectors[:,idx])
                    
            # compute S
            tmp_val = np.sqrt(values)            
            l = len(idx)
            self.S = scipy.sparse.spdiags(tmp_val, 0, l, l,format='csc') 
            
            # and the inverse of it            
            S_inv = scipy.sparse.spdiags(1.0/tmp_val, 0, l, l,format='csc')
            
            # compute V from it
            self.V = self.U.transpose() * self.data
            self.V = S_inv * self.V
    
        def _sparse_left_svd():        
            # for some reasons arpack does not allow computation of rank(A) eigenvectors (??)
            AA = self.data.transpose()*self.data
    
            if self.data.shape[1] > 1:                
                # do not compute full rank if desired
                if self._k > 0 and self._k < AA.shape[1]-1:
                    k = self._k
                else:
                    k = self.data.shape[1]-1
                
                values, v_vectors = linalg.eigsh(AA,k=k)                    
            else:                
                values, v_vectors = eigh(AA.todense())    
           
            # get rid of negative/too low eigenvalues   
            s = np.where(values > self._EPS)[0]
            v_vectors = v_vectors[:, s] 
            values = values[s]
            
            # sort eigenvectors according to largest value
            idx = np.argsort(values)[::-1]                  
            values = values[idx]
            
            # argsort sorts in ascending order -> access is backwards            
            self.V = scipy.sparse.csc_matrix(v_vectors[:,idx])      
            
            # compute S
            tmp_val = np.sqrt(values)            
            l = len(idx)      
            self.S = scipy.sparse.spdiags(tmp_val, 0, l, l,format='csc') 
            
            # and the inverse of it                                         
            S_inv = scipy.sparse.spdiags(1.0/tmp_val, 0, l, l,format='csc')
            
            self.U = self.data * self.V * S_inv        
            self.V = self.V.transpose()           
        
        if self._rows >= self._cols:
            #if scipy.sparse.issparse(self.data):
            #    _sparse_left_svd()
            #else:
            #    _left_svd()
            _left_svd()
        else:
            # if scipy.sparse.issparse(self.data):
            #     _sparse_right_svd()
            # else:
            #     _right_svd()
            _right_svd()

def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
