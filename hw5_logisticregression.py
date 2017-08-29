import numpy as np
import matplotlib.pyplot as plt
import random

d = 2
N = 100

def create_dataset(size=N):
    return np.random.uniform(-1,1,(size,d))

def create_targets():
    w = np.random.uniform(-1,1,(1,d+1))[0]
    data = create_dataset()
    y = calc_target(w,data)
    return w, (data,y)

def calc_target(w, data):
    return np.sign(w.dot(add_bias(data).T))

def add_bias(X):
    return np.hstack((np.ones((X.shape[0],1)), X))

def verify_data(g, data):
    X = data[0]
    y = data[1]
    
    plt.scatter(X[y>0][:,0],X[y>0][:,1],color="red")
    plt.scatter(X[y<0][:,0],X[y<0][:,1],color="blue")
    
    x2_intercept = -g[0]/g[2]
    x1_intercept = -g[0]/g[1]

    slope,intercept = np.polyfit([0,x1_intercept],[x2_intercept,0],1)
    x_intervals = np.arange(-1,1.1,.2)
    y_intervals = x_intervals*slope + intercept
    plt.plot(x_intervals,y_intervals)
    
    plt.axhline(y=0,color="gray")
    plt.axvline(x=0,color="gray")
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    
    plt.show()
    
def sigmoid(x):
    return 1/(1+np.e**-x)

class LogisticRegression():
    def __init__(self):
        self.w = np.zeros(d+1)
        self.eta = .01
        self.threshold = .01
    
    def fit(self,data,target):
        X = add_bias(data)
        data = list(zip(X,target))
        
        for i in range(1,10001):    
            random.shuffle(data)
            w1 = self.w.copy()
            for x_i, y_i in data:
                dE = -(y_i * x_i) / (1 + np.e**(y_i*self.w.dot(x_i)))
                self.w -= self.eta*dE

            w_score = np.sqrt(sum((w1 - self.w)**2))    
            if w_score < self.threshold:
                return i
        return -1
    
    def predict(self,data):
        X = add_bias(data)
        return sigmoid(self.w.dot(X.T))

    def calc_cross_entropy(self,data,y):
        X = add_bias(data)
        return np.log(1 + np.e**(-y * self.w.dot(X.T))).mean()

num_epochs = []
errors = []
num_iter=100
for i in range(1,num_iter+1):
    print("Iteration: [ {} / {} ]".format(i,num_iter))
    target_function, target_data = create_targets()
    #verify_data(target_function,target_data)
    
    lr = LogisticRegression()
    num_epoch = lr.fit(target_data[0],target_data[1])
    
    new_X = create_dataset(10000)
    new_y = calc_target(target_function, new_X)
    error = lr.calc_cross_entropy(new_X,new_y)
    
    num_epochs.append(num_epoch)
    errors.append(error)
#    break

print("Average number of epochs: %d" % np.mean(num_epochs))
print("Average cross entropy error: %.3f" % np.mean(errors))