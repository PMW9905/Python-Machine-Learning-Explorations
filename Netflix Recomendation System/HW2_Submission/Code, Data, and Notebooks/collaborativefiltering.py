import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
import numpy.ma as ma
import math
import gc


class MemoryBasedCollaborativeFiltering:
    
    def __init__(self):
        self.V_diff = None
        self.V_avg = None
        self.W = None
        self.weighted_diff = None
    
    def fit(self, V):
        V_sparse = sp.csr_matrix(V)
        self.V_sum = V_sparse.sum(axis=1)
        print('sum')
        self.V_count = np.array([row.count_nonzero() for row in V_sparse])
        print('count')
        self.V_avg = np.array([self.V_sum[i]/self.V_count[i] for i in range(self.V_sum.shape[0])])
        self.V_avg = self.V_avg.flatten()
        print('avg')

        # Make array of ones from V_sparse 
        self.empty_V_sparse = V_sparse.copy()
        print('copy')
        self.empty_V_sparse[self.empty_V_sparse != 0] = 1
        print('setzeros')
        # Make array of diagonals from self.V_avg
        self.V_avg_sparse = sp.diags(self.V_avg,format='csr')
        print('diags')
        # Multiply
        self.placement = self.V_avg_sparse.dot(self.empty_V_sparse)
        print('placement')
        # Subtract V_sparse from that to get average.
        self.diff = V_sparse - self.placement
        print('diff')

        self.numerator = self.diff.dot(self.diff.T)

        # -----------------------------------------------------------------

        # Create a matrix of squared diff values
        self.diff_squared = self.diff.power(2)
        print('power')
        # Create a matrix of 1s and 0s in the shape of diff
        self.diff_empty = self.diff.copy()
        self.diff_empty[self.diff_empty!=0] = 1
        print('diff empty')
        # This 0 1 matrix allows us to sum up all values of a row 
        # but only if values are shared between the column it's being
        # multiplied with.
        self.diff_squared_sums = self.diff_squared.dot(self.diff_empty.T)
        print('diff square sum')
        # Now that we have these sums, just add.
        self.denom = self.diff_squared_sums.multiply(self.diff_squared_sums.T)
        print('initial denom')
        self.denom = self.denom.sqrt()
        print('sqrt')
        np.reciprocal(self.denom.data, out=self.denom.data)
        print('reciprocal')
        self.W = self.numerator.multiply(self.denom)
        print('created W')

        # -----------------------------------------------------------------

    def predict(self, X, Y):
        MAE = 0
        RMSE = 0
        n = len(Y)
        for i in range(n):
            err = Y[i] - self.p(X[i,1],X[i,0])
            MAE+=abs(err)
            RMSE+=err**2
        MAE = MAE/n
        RMSE = math.sqrt(RMSE / n)
        return MAE, RMSE

    def p(self, a, j):
        v_a = self.V_avg[a]
        k = np.sum(np.abs(self.W[a]))
        if k != 0:
            k = 1 / k
        weighted_sum = self.W[a].dot(self.diff[:,j])
        return v_a + (k * weighted_sum[0,0])