import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def target_function(x):
    return np.sin(x*np.pi)

calc_y = np.vectorize(target_function)
X = np.arange(-1,1.1,.1)
plt.plot(X,calc_y(X))

coefs = []
var = []
for i in range(1000):
    X = np.random.uniform(-1,1,(2,1))
#    X = X ** 2 # x^2
    y = calc_y(X)
    
#    model = LinearRegression(fit_intercept=False) # y = ax
    model = LinearRegression(fit_intercept=True) # y = ax + b
    model.fit(X,y)
    
    predictions = model.predict(X)
    plt.plot(X,predictions,color="orange",alpha=.1)
    coefs.append(model.coef_[0][0])
    var.append([X, predictions])

slope = np.mean(coefs)
X = np.arange(-1,1.1,.01)
hypothesis = X*slope
plt.plot(X,hypothesis,color="red")
plt.show()

bias = np.mean((hypothesis - calc_y(X))**2)

var = np.array(var)
X = var[:,0].flatten()
y = var[:,1].flatten()

variance = np.mean((y - X*slope)**2)

print("Slope: %2f" % slope)
print("Bias: %2f" % bias)
print("Variance: %2f" % variance)