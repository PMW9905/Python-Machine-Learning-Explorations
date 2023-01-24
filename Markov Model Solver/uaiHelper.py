import numpy as np
from dataclasses import dataclass

@dataclass
class UAIFunction:
    # Number of variables participating in the function
    size: int 
    # List of variables s.t. that the index is the order of the variables.
    indexes: list() 
    # List of values for each permutation of variable values.
    values: list() 

class UAIHelper:
    def __init__(self, fpath):
        with open(fpath,'r') as f:
            # Ignore type; will always be markov, for our case.
            f.readline()
            self.num_vars = int(f.readline())
            # according to prof, variables will always be binary, so we can ignore 3rd line.
            f.readline()

            # Ignore whitespace line.
            f.readline()
            self.num_functions = int(f.readline())
            # Each index contains a set! 
            self.var_function_participation = np.array([set() for _ in range(self.num_vars)])
            self.functions = np.array([UAIFunction(0,[],[]) for _ in range(self.num_functions)]).astype(type(UAIFunction(0,[],[])))

            for i in range(self.num_functions):
                line = f.readline().replace('\n',' ').split(' ')
                vars = int(line[0])
                indexes = []
                for j in range(vars):
                    indexes.append(int(line[1+j]))
                    self.var_function_participation[indexes[j]].add(i)
                self.functions[i].size = vars
                self.functions[i].indexes = indexes
            
            f.readline()

            for i in range(self.num_functions):
                line = f.readline().replace('\n',' ').split(' ')
                num_values = int(line[0])
                values = []
                for j in range(num_values):
                    values.append(float(line[1+j]))
                self.functions[i].values = values

    def createDataMatrix(self, X, Q_size, X_indexes, Q_indexes):

        # create set of all included X indexes.
        all_x_indexes = set()
        for x_i in X_indexes:
            all_x_indexes.add(x_i)

        # create array of var name to index
        name_to_index = np.zeros(self.num_vars).astype(int)
        for index, value in enumerate(X_indexes):
            name_to_index[value] = index

        # create new matrix of size (x.shape[0], x.shape[1]+q.shape[1])
        new_X = np.zeros((X.shape[0],X.shape[1]+Q_size),dtype=float)
        # for each row:
        for i in range(X.shape[0]):
            # fill in all begining entries with x.
            for j in range(X.shape[1]):
                new_X[i,j] = X[i,j]

            # for each q in Q:
            for j in range(Q_size):
                q_index = Q_indexes[j]
                # for every function q is a part of:
                weight_true = 1
                weight_false = 1
                for function_index in self.var_function_participation[q_index]:
                    pass
                    # calculate weight multiplication sum of true & false
                    current_function = self.functions[function_index]

                    variable_table_index = 0
                    q_index_modifier = 0

                    # Find the index of the false state & the modifier to get the true state.
                    for counter, f_variable_index in enumerate(current_function.indexes):
                        # if all variable participants present, use; otherwise, ignore function.
                        if f_variable_index not in all_x_indexes and f_variable_index != q_index:
                            continue # if you don't have sufficient metrix to calculate, ignore function
                        if f_variable_index != q_index and X[i,name_to_index[int(f_variable_index)]] == 1:
                            variable_table_index += len(current_function.values) / 2**(counter+1)
                        elif f_variable_index == q_index:
                            q_index_modifier = len(current_function.values) / 2**(counter+1)
                    
                    # then calculate the true and false weights for q.
                    weight_true*=current_function.values[int(variable_table_index + q_index_modifier)]
                    weight_false*=current_function.values[int(variable_table_index)]
                # Calculate weight_true / (weight_true + weight_false)
                weight_final = weight_true / (weight_true + weight_false)
                # Add this to the matrix.
                new_X[i,j + X.shape[1]] = weight_final
        
        # return new matrix and use!
        return new_X
