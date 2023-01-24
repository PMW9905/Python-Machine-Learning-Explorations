import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import glob
import os
import re
from random import shuffle

class MCAPLogisticRegression():
    def __init__(self):
        pass

    def train(self, x_df, y_df, _lambda, learning_rounds):
        x_np = x_df.to_numpy()
        x_np = np.insert(x_np,0,np.ones(x_np.shape[0]),axis=1)
        y_np = y_df.astype('int32').to_numpy()

        num_rows = x_np.shape[0]
        num_x_vars = x_np.shape[1]

        learning_rate = .00008

        self.W = np.zeros(num_x_vars)
        w_delta = np.ones(num_x_vars)
        w_delta_goal = .0004 * np.ones(num_x_vars)

        for _ in range(learning_rounds):
            z = np.dot(x_np,self.W)
            error=np.zeros(num_rows)
            for i in range(len(z)):
                exp = math.exp(z[i])
                sig_fun = (exp / (1 + exp))
                error[i] = y_np[i][0] - sig_fun
            w_delta = learning_rate*np.dot(error,x_np) - (learning_rate*_lambda*self.W)
            if np.all(abs(w_delta) <= w_delta_goal):
                break
            self.W = self.W + w_delta 

    def test(self, x_df):
        x_np = x_df.to_numpy()
        
        z = np.dot(x_np,self.W[1:])
        predictions=np.zeros_like(z)
        for i in range(len(z)):
            exp = math.exp(self.W[0] + z[i])
            predictions[i] = (exp / (1 + exp))
        return np.where(predictions >= .5, True, False)

    def findOptimalLambda(df):
        # Splitting dataset into test and train.
        df_train, df_test = train_test_split(df,train_size=.7)
        
        # Splitting test and train into their respective x and y sets.
        y_train, y_test = df_train.iloc[:,0].to_frame(), df_test.iloc[:,0].to_frame()
        x_train, x_test = df_train.iloc[:,1:], df_test.iloc[:,1:]
        
        # Converting types from string to boolean
        y_train['row_class'] = np.where(y_train['row_class'] == 'ham', False, True)
        y_test['row_class'] = np.where(y_test['row_class'] == 'ham', False, True)

        # Setting y_test to numpy for easier indexing.
        best_lambda = [.1,0]
        for _lambda in [.1,.01,.001]:
            logReg = MCAPLogisticRegression()
            logReg.train(x_train,y_train,_lambda,50)
            logReg_results = logReg.test(x_test)
            result_matrix = logReg.buildResultsMatrix(logReg_results,y_test)

            quality = sum(result_matrix.values())

            if best_lambda[1] < quality:
                best_lambda = [_lambda, quality]
        return best_lambda[0]

    def buildResultsMatrix(self, results, df_y):
        np_y = df_y.to_numpy().flatten()
        result_matrix = {'TP':0,'FP':0,'TN':0,'FN':0}
        for i in range(len(results)):
            if results[i] == True:
                if np_y[i] == True:
                    result_matrix['TP']+=1
                else:
                    result_matrix['FP']+=1
            else:
                if np_y[i] == False:
                    result_matrix['TN']+=1
                else:
                    result_matrix['FN']+=1
        return result_matrix 

