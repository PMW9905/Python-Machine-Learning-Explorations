import numpy as np


class Data:
  #fpath: File path of the .data file
  
  #self.evid_var_ids: Contains the indices of the observed variables
  #self.query_var_ids: Contains the indices of the query variables
  #self.hidden_var_ids: Contains the indices of the hidden variables
  
  #self.evid_assignments: Assignments to evid variables
  #self.query_assignments: Assignments to query variables
  #self.weights: Pr(e, q)
  def __init__(self, fpath):

    f = open(fpath, "r")
    
    self.nvars = int(f.readline()) #1
    
    line = np.asarray(f.readline().split(), dtype=np.int32)#2
    self.evid_var_ids = line[1:]
    evid_indices = range(1, self.evid_var_ids.shape[0]*2, 2)

    line = np.asarray(f.readline().split(), dtype=np.int32) #3
    self.query_var_ids = line[1:]
    query_indices = range(self.evid_var_ids.shape[0]*2+1, (self.evid_var_ids.shape[0]+self.query_var_ids.shape[0])*2, 2)

    line = np.asarray(f.readline().split(), dtype=np.int32)#4
    self.hidden_var_ids = line[1:]
    
    line = f.readline()#5
    self.nproblems = int(f.readline())#6
    
    self.evid_assignments = []
    self.query_assignments = []
    self.weights = []
    for i in range(self.nproblems):
      line = np.asarray(f.readline().split(), dtype=float)
      self.evid_assignments.append(np.asarray(line[evid_indices], dtype=np.int32))
      self.query_assignments.append(np.asarray(line[query_indices], dtype=np.int32))
      self.weights.append(line[-1])
    self.evid_assignments = np.asarray(self.evid_assignments)
    self.query_assignments = np.asarray(self.query_assignments)
    self.weights = np.asarray(self.weights)

  def convertToXY(self):
    return (self.evid_assignments, self.query_assignments)

  def convertResults(self, query_predictions):
    out = np.zeros((query_predictions.shape[0], 1+2*self.query_var_ids.shape[0]), dtype=int)
    out[:, 2::2] = query_predictions[:, :]
    out[:, 1::2] = self.query_var_ids
    out[:, 0] = self.query_var_ids.shape[0]
    return out