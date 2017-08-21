import numpy as np
import string
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

N = 1000
d = 2

def calc_y(X):
    return np.sign(np.square(X).sum(axis=1) - .6)

def add_noise(y):
    mask = np.random.choice([False,True],len(y),p=[.9,.1])
    y[mask] = y[mask]*-1
    return y

def calc_ys_hypotheses(X):
#    x0 = X[:,0]
#    x1 = X[:,1]
#    a = np.sign(-1 - .05*x0 + .08*x1 + .13*x0*x1 + 1.5*x0**2 + 1.5*x1**2)
#    b = np.sign(-1 - .05*x0 + .08*x1 + .13*x0*x1 + 1.5*x0**2 + 15*x1**2)
#    c = np.sign(-1 - .05*x0 + .08*x1 + .13*x0*x1 + 15*x0**2 + 1.5*x1**2)
#    d = np.sign(-1 - 1.5*x0 + .08*x1 + .13*x0*x1 + .05*x0**2 + .05*x1**2)
#    e = np.sign(-1 - .05*x0 + .08*x1 + 1.5*x0*x1 + .15*x0**2 + .15*x1**2)
    
    #polyfeature order is 1, x0, x1, x0^2, x0x1, x1^2
    w_hypos = np.array([
            [-1, -.05, .08, 1.5, .13, 1.5],
            [-1, -.05, .08, 1.5, .13, 15],
            [-1, -.05, .08, 15, .13, 1.5],
            [-1, -1.5, .08, .05, .13, .05],
            [-1, -.05, .08, .15, 1.5, .15],
            ])
    
    return np.sign(X.dot(w_hypos.T))
#    return a,b,c,d,e

def generate_data():
    return np.random.uniform(-1,1,(N,d))

poly = PolynomialFeatures(2)

percent_misclassified = []
percent_misclassified_oos = []
percent_correct = []
for i in range(1000):
    X = generate_data()
    y = calc_y(X)
    add_noise(y)
    
    lr = LinearRegression()
    lr.fit(X,y)
    prediction = np.sign(lr.predict(X))
    percent_misclassified.append(np.sum(prediction!=y)/len(prediction))
    
    X_nonlinear = poly.fit_transform(X)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_nonlinear,y)
    prediction = np.sign(lr.predict(X_nonlinear))
    predictions = calc_ys_hypotheses(X_nonlinear)
    outputs = []
    for i in range(predictions.shape[1]):
        outputs.append(np.sum(prediction==predictions[:,i])/len(prediction))
    percent_correct.append(outputs)
    
    X_oos = generate_data()
    X_oos_nonlinear = poly.fit_transform(X_oos)
    y_oos = calc_y(X_oos)
    add_noise(y_oos)
    prediction_a = calc_ys_hypotheses(X_oos_nonlinear)[:,0]
    percent_misclassified_oos.append(np.sum(prediction_a!=y_oos)/len(prediction))
    
print("Average percent misclassified: %.2f" % np.mean(percent_misclassified))

percent_correct = np.mean(np.array(percent_correct),axis=0)
for h, p in zip(list(string.ascii_lowercase),percent_correct):
    print("Average percent misclassified ({}): {:.2f}".format(h,p))

print("Average percent misclassified (out of sample): %.2f" % np.mean(percent_misclassified_oos))
