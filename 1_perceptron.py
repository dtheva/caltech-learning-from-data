import numpy as np
import random

d=2
N=100
NUM_ITER=1000

class PLA:
    def __init__(self):
        self.w = np.zeros(d+1) # +1 for bias
        self.max_iter = 10000
        
    def get_not_equal(self,a1,a2):
        return list(np.nonzero(a1-a2)[0])
    
    def train(self,data,target):
        bias = np.ones(data.shape[0]).reshape(-1,1)
        X = np.hstack([bias,data])
        for num_iter in range(1,self.max_iter+1):
            prediction = np.sign(self.w.dot(X.T))
            delta = self.get_not_equal(prediction,target)

            if not delta:
                return num_iter
            else:
                misclassified_point = random.choice(delta)

                x_n = X[misclassified_point]
                y_n = y[misclassified_point]
                
                self.w += x_n*y_n
        return -1
    
    def predict(self,data):
        bias = np.ones(data.shape[0]).reshape(-1,1)
        X = np.hstack([bias,data])
        return np.sign(self.w.dot(X.T))

def create_target_function(X):
    bias = np.ones(N).reshape(-1,1)
    random_weight = np.random.uniform(-1,1,d+1)
    target_X = np.hstack([bias,X])
    return random_weight, np.sign(random_weight.dot(target_X.T))

def predict(w, data):
    bias = np.ones(data.shape[0]).reshape(-1,1)
    X = np.hstack([bias,data])
    return np.sign(w.dot(X.T))
    
iter_to_converge = []
p_fx_not_gx = []
for i in range(NUM_ITER):
    X = np.random.uniform(-1,1,(N,d))
    fx, y = create_target_function(X)
    
    pla = PLA()
    iter_to_converge.append(pla.train(X,y))
    
    num_sample = 10000
    X_test = np.random.uniform(-1,1,(num_sample,d))
    y_predict = pla.predict(X_test)
    y_target = predict(fx,X_test)

    p_fx_not_gx.append(np.sum(y_predict != y_target)/num_sample)
        
print("Average iter to converge: %d" % np.mean(iter_to_converge))
print("Average prob f(x) != g(x): %2f" % np.mean(p_fx_not_gx))