import numpy as np
import matplotlib.pyplot as plt
import random

d = 2
N = 100

def create_dataset(size=N):
    return np.random.uniform(-1,1,(size,d))

def create_targets(plot=False):
    line_coords = np.random.uniform(-1,1,(2,d))
    slope,intercept = np.polyfit(line_coords[:,0],line_coords[:,1],1)

    data = create_dataset()
    y = calc_target((slope,intercept),data)
    
    if plot:
        x_coords = np.array([-1,1])
        y_coords = x_coords*slope + intercept
        plt.plot(x_coords,y_coords) 
        plt.scatter(data[y>0][:,0],data[y>0][:,1],color="red")
        plt.scatter(data[y<0][:,0],data[y<0][:,1],color="blue")
        
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.show()

    return (slope,intercept), (data,y)    

def calc_target(line, data):
    slope = line[0]
    intercept = line[1]
    x0 = data[:,0]
    y0 = data[:,1]
    
    y_out = np.zeros(x0.shape[0])
    y_out[x0*slope + intercept > y0] = 1
    y_out[x0*slope + intercept < y0] = -1

    return y_out

def add_bias(X):
    return np.hstack((np.ones((X.shape[0],1)), X))
    
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
    target_function, target_data = create_targets(plot=False)
    
    lr = LogisticRegression()
    num_epoch = lr.fit(target_data[0],target_data[1])
    
    new_X = create_dataset(100)
    new_y = calc_target(target_function, new_X)
    error = lr.calc_cross_entropy(new_X,new_y)
    
    num_epochs.append(num_epoch)
    errors.append(error)

print("Average number of epochs: %d" % np.mean(num_epochs))
print("Average cross entropy error: %.3f" % np.mean(errors))