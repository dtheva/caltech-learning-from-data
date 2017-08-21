import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from hw1_perceptron import PLA

d = 2
N = 100
N_PLA = 10
check = False

def add_bias(data):
    return np.hstack([np.ones(data.shape[0]).reshape(-1,1),data])

#class LinearRegression:
#    def __init__(self):
#        self.w = np.zeros(d+1)
#        self.max_iter = 10000
#
#    def calc_error(y1,y2):
#        return (y1-y2)**2
#    
#    def train(self,data):
#        X = add_bias(data)
#        prediction = calc_output(self.w,X)
#        print(prediction)
##        for i in range(max_iter):
        
def calc_output(w,X):
    return np.sign(w.dot(X.T))

def get_targets(data):
    target_function = np.random.uniform(-1,1,d+1)
    X = add_bias(data)
    target_output = calc_output(target_function,X)
    return target_function, target_output

def check_scatter(X,y):
    plt.scatter(X[y==1][:,0],X[y==1][:,1],color="r")
    plt.scatter(X[y!=1][:,0],X[y!=1][:,1],color="b")
    plt.show()    

percent_misclassified = []        
percent_misclassified_oos = []
num_iter_pla = []
for i in range(1000):
    X = np.random.uniform(-1,1,(N,d))
    target_function, y = get_targets(X)
    if check:
        check_scatter(X,y)
    
    lr = LinearRegression()
    lr.fit(X,y)
    prediction = np.sign(lr.predict(X))
    percent_misclassified.append(np.sum(prediction!=y)/len(prediction))
    
    X_oos = np.random.uniform(-1,1,(1000,d))
    y_oos = calc_output(target_function,add_bias(X_oos))
    prediction_oos = np.sign(lr.predict(X_oos))
    percent_misclassified_oos.append(np.sum(prediction_oos!=y_oos)/len(prediction_oos))
    
    lr_w = np.insert(lr.coef_,0,lr.intercept_)
    pla = PLA(lr_w)
    
    X_pla = np.random.uniform(-1,1,(N_PLA,d))
    y_pla = calc_output(target_function,add_bias(X_pla))
    num_iter_pla.append(pla.train(X_pla,y_pla))
    
print("Average percent misclassified: %.2f" % np.mean(percent_misclassified))
print("Average percent misclassified (out of sample): %.2f" % np.mean(percent_misclassified_oos))
print("Average iterations to converge: %d" % np.mean(num_iter_pla))