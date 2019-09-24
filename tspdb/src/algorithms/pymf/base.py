# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF base class used in (almost) all matrix factorization methods

"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from numpy.linalg import eigh
from scipy.misc import factorial

__all__ = ["PyMFBase", "PyMFBase3", "eighk", "cmdet", "simplex"]
_EPS = np.finfo(float).eps

def eighk(M, k=0):
    """ Returns ordered eigenvectors of a squared matrix. Too low eigenvectors
    are ignored. Optionally only the first k vectors/values are returned.

    Arguments
    ---------
    M - squared matrix
    k - (default 0): number of eigenvectors/values to return

    Returns
    -------
    w : [:k] eigenvalues 
    v : [:k] eigenvectors

    """
    values, vectors = eigh(M)            
              
    # get rid of too low eigenvalues
    s = np.where(values > _EPS)[0]
    vectors = vectors[:, s] 
    values = values[s]                            
             
    # sort eigenvectors according to largest value
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:,idx]

    # select only the top k eigenvectors
    if k > 0:
        values = values[:k]
        vectors = vectors[:,:k]

    return values, vectors


def cmdet(d):
    """ Returns the Volume of a simplex computed via the Cayley-Menger
    determinant.

    Arguments
    ---------
    d - euclidean distance matrix (shouldn't be squared)

    Returns
    -------
    V - volume of the simplex given by d
    """
    D = np.ones((d.shape[0]+1,d.shape[0]+1))
    D[0,0] = 0.0
    D[1:,1:] = d**2
    j = np.float32(D.shape[0]-2)
    f1 = (-1.0)**(j+1) / ( (2**j) * ((factorial(j))**2))
    cmd = f1 * np.linalg.det(D)

    # sometimes, for very small values, "cmd" might be negative, thus we take
    # the absolute value
    return np.sqrt(np.abs(cmd))


def simplex(d):
    """ Computed the volume of a simplex S given by a coordinate matrix D.

    Arguments
    ---------
    d - coordinate matrix (k x n, n samples in k dimensions)

    Returns
    -------
    V - volume of the Simplex spanned by d
    """
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0]+1, d.shape[1]))
    D[1:,:] = d
    V = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return V


class PyMFBase():
    """
    PyMF Base Class. Does nothing useful apart from poviding some basic methods.
    """
    # some small value
   
    _EPS = _EPS
    
    def __init__(self, data, num_bases=4, **kwargs):
        """
        """
        
        def setup_logging():
            # create logger       
            self._logger = logging.getLogger("pymf")
       
            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()
        
        # set variables
        self.data = data       
        self._num_bases = num_bases             
      
        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape
        

    def residual(self):
        """ Returns the residual in % of the total amount of data

        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0*res/np.sum(np.abs(self.data))
        return total
        
    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.

        Returns:
        -------
        frobenius norm: F = ||data - WH||

        """
        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ))            
        else:
            err = None

        return err
        
    def _init_w(self):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as 
        # they have difficulties recovering from zero.
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10**-4
        
    def _init_h(self):
        """ Initalize H to random values [0,1].
        """
        self.H = np.random.random((self._num_bases, self._num_samples)) + 10**-4
        
    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """ 
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean 
        """
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data
        
        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].
        
        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """
        
        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)        
        
        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self,'W') and compute_w:
            self._init_w()
               
        if not hasattr(self,'H') and compute_h:
            self._init_h()                   
        
        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)
             
        for i in xrange(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()                                        
         
            if compute_err:                 
                self.ferr[i] = self.frobenius_norm()                
                self._logger.info('FN: %s (%s/%s)'  %(self.ferr[i], i+1, niter))
            else:                
                self._logger.info('Iteration: (%s/%s)'  %(i+1, niter))
           

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


class PyMFBase3():    
    """      
    PyMFBase3(data, show_progress=False)
    
    Base class for factorizing a data matrix into three matrices s.t. 
    F = | data - USV| is minimal (e.g. SVD, CUR, ..)
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV                
    
    """
    _EPS = _EPS

    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        """
        """
        self.data = data
        (self._rows, self._cols) = self.data.shape

        self._rrank = self._rows
        if rrank > 0:
            self._rrank = rrank
            
        self._crank = self._cols
        if crank > 0:            
            self._crank = crank
        
        self._k = k
    
    def frobenius_norm(self):
        """ Frobenius norm (||data - USV||) for a data matrix and a low rank
        approximation given by SVH using rank k for U and V
        
        Returns:
            frobenius norm: F = ||data - USV||
        """    
        if scipy.sparse.issparse(self.data):
            err = self.data - (self.U*self.S*self.V)
            err = err.multiply(err)
            err = np.sqrt(err.sum())
        else:                
            err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
            err = np.sqrt(np.sum(err**2))
                            
        return err
        
    
    def factorize(self):    
        pass

def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
