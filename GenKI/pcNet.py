import numpy as np
from numpy.linalg import svd
from scipy import sparse
import os
import time

try:
    import ray
except ImportError:
    class dummy_ray():
        def __init__(self):
            self.remote = lambda x:0
            print("ray not initialized")
        def is_initialized(self):
            return False   
        def init(self, num_cpus):
            self.num_cpu = num_cpus 
        def put(self, x):
            self.x = x
    ray = dummy_ray()


def pcCoefficients(X, K, nComp):
    y = X[:, K] 
    Xi = np.delete(X, K, 1)
    U, s, VT = svd(Xi, full_matrices=False) 
    #print ('U:', U.shape, 's:', s.shape, 'VT:', VT.shape)
    V = VT[:nComp, :].T
    #print('V:', V.shape)
    score = Xi@V
    t = np.sqrt(np.sum(score**2, axis=0))
    score_lsq = ((score.T / (t**2)[:, None])).T
    beta = np.sum(y[:, None]*score_lsq, axis=0)
    beta = V@beta
    return list(beta) 


def pcNet(X, # X: cell * gene
    nComp: int = 3, 
    scale: bool = True, 
    symmetric: bool = True, 
    q: float = 0., # q: 0-100
    as_sparse: bool = True,
    random_state: int = 0): 
    X = X.toarray() if sparse.issparse(X) else X
    if nComp < 2 or nComp >= X.shape[1]:
        raise ValueError("nComp should be greater or equal than 2 and lower than the total number of genes") 
    else:
        np.random.seed(random_state)
        n = X.shape[1] # genes    
        B = np.array([pcCoefficients(X, k, nComp) for k in range(n)])    
        A = np.ones((n, n), dtype=float)
        np.fill_diagonal(A, 0)
        for i in range(n):
            A[i, A[i, :]==1] = B[i, :]                
        if scale:
            absA = abs(A)
            A = A / np.max(absA)
        if q > 0:
            A[absA < np.percentile(absA, q)] = 0
        if symmetric: # place in the end
            A = (A + A.T)/2
        #diag(A) <- 0
        if as_sparse:
            A = sparse.csc_matrix(A)        
        return A


@ray.remote
def pc_net_parallel(X,  # X: cell * gene
                    nComp: int = 3,
                    scale: bool = True,
                    symmetric: bool = True,
                    q: float = 0.,
                    as_sparse: bool = True,
                    random_state: int = 0):
    return pcNet(X, nComp = nComp, scale = scale, symmetric = symmetric, q = q, 
                 as_sparse = as_sparse, random_state = random_state)


def pc_net_single(X,  # X: cell * gene
                    nComp: int = 3,
                    scale: bool = True,
                    symmetric: bool = True,
                    q: float = 0.,
                    as_sparse: bool = True,
                    random_state: int = 0):
    return pcNet(X, nComp = nComp, scale = scale, symmetric = symmetric, q = q, 
                 as_sparse = as_sparse, random_state = random_state)


def make_pcNet(X, 
          nComp: int = 3, 
          scale: bool = True, 
          symmetric: bool = True, 
          q: float = 0., 
          as_sparse: bool = True, 
          random_state: int = 0,
          n_cpus: int = 1, # -1: use all CPUs
          timeit: bool = True):
    start_time = time.time()
    if n_cpus != 1:
        if ray.is_initialized():
            ray.shutdown()
        if n_cpus == -1:
            n_cpus = os.cpu_count()
        ray.init(num_cpus = n_cpus)
        print(f"ray init, using {n_cpus} CPUs")

        X_ray = ray.put(X) # put X to distributed object store and return object ref (ID)
        # print(X_ray)
        net = pc_net_parallel.remote(X_ray, nComp = nComp, scale = scale, symmetric = symmetric, q = q, 
                    as_sparse = as_sparse, random_state = random_state)
        net = ray.get(net)
    else:
        net = pc_net_single(X, nComp = nComp, scale = scale, symmetric = symmetric, q = q, 
            as_sparse = as_sparse, random_state = random_state)
    if ray.is_initialized():
        ray.shutdown()
    if timeit:
        duration = time.time() - start_time
        print("execution time of making pcNet: {:.2f} s".format(duration))
    return net


def main():
    counts = np.random.randint(0, 10, (5, 100))
    net = make_pcNet(counts, as_sparse = True, timeit = True, n_cpus = 1)
    print(f"input counts shape: {counts.shape},\nmake pcNet completed, shape: {net.shape}")
    return 100


if __name__ == "__main__":
    import sys
    sys.exit(main())
    