#Learning from Data HW3: brute force breakpoints

import numpy as np
import itertools
import random

AXIS_MIN = 0
AXIS_MAX = 10

#very basic PLA, so possible that it doesn't converge when points are near multicollinearity
class PLA():
    def __init__(self,dim):
        self.w = np.zeros(dim+1)
        self.max_iter = 1000
    
    def add_bias(self,data):
        return np.hstack([np.ones(data.shape[0]).reshape(-1,1),data])
    
    def get_misclassified(self,prediction,actual):
        return list(np.nonzero(prediction-actual)[0])
    
    #returns true if converged, else false
    def train(self,data,y):
        X = self.add_bias(data)
        for i in range(self.max_iter):
            output = np.sign(X.dot(self.w))
            misclassified = self.get_misclassified(output,y)            
            if misclassified:
                n = random.choice(misclassified)
                X_n = X[n]
                y_n = y[n]
                self.w += X_n * y_n
            else:
                return True
        return False

def generate_data(num_points,dim):
    data = np.random.randint(AXIS_MIN,AXIS_MAX,(num_points,dim))
    unique_data = set([tuple(x) for x in data])
    while len(unique_data) != len(data):
        data = np.random.randint(AXIS_MIN,AXIS_MAX,(num_points,dim))
        unique_data = set([tuple(x) for x in data])
    return data

#outputs can be -1 or 1
def generate_possible_dichotomies(num_points):
    outputs = []
    for combination in itertools.product([-1,1],repeat=num_points):
        outputs.append(np.array(combination))
    return outputs

def check_shatterable(data,output):
    pla = PLA(dim)
    if pla.train(data,output):
        return True
    else:
        return False

max_N = 10 # max # of points
dim = 3 # dimensionality of points
num_datasets = 10

for N in range(3,max_N+1):
    possible_dichotomies = generate_possible_dichotomies(N)
    
    breakpoint = True
    for i in range(num_datasets): 
        data = generate_data(N,dim)
        
        shatterable = True
        for dichotomy in possible_dichotomies:
            if not check_shatterable(data,dichotomy):
                shatterable = False
                break
            
        if shatterable:
            breakpoint = False
            break
        
    if breakpoint:
        print("Breakpoint at {} points in {} dimensions".format(N,dim))
        break