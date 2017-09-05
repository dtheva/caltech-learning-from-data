import numpy as np
import matplotlib.pyplot as plt

def nonlinear_transform(data):
    x1 = np.ones(data.shape[0]).reshape(-1,1)
    x2 = data[:,0].reshape(-1,1)
    x3 = data[:,1].reshape(-1,1)
    x4 = x2**2
    x5 = x3**2
    x6 = x2*x3
    x7 = np.abs(x2-x3)
    x8 = np.abs(x2+x3)
    x9 = data[:,-1].reshape(-1,1)

    return np.hstack([x1,x2,x3,x4,x5,x6,x7,x8,x9])

def calc_linear_weights(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
def predict(X,w):
    return np.sign(X.dot(w))

def summarize_results(predictions, actual, k):
    print("[ {} / {} ] misclassified for (k={})".format(
            len(predictions[predictions!=actual]),len(actual),k))
    
def linear_regression(X_train,y_train,X_validate,y_validate):
    weights = {}
    for k in range(3,8):
        X_k = X_train[:,:k+1]
        w = calc_linear_weights(X_k,y_train)
        weights[k] = w
    
        X_k_validate = X_validate[:,:k+1]
        predictions = predict(X_k_validate,w)
        summarize_results(predictions, y_validate, k)    

    return weights

def verify_regression(X_test, y_test, weights):
    for k, w in weights.items():
        X_k = X_test[:,:k+1]
        predictions = predict(X_k,w)
        summarize_results(predictions, y_test, k)

train = np.fromfile("in.dta",sep=" ").reshape(-1,3)
test = np.fromfile("out.dta",sep=" ").reshape(-1,3)

train_nonlinear = nonlinear_transform(train)

X_train = train_nonlinear[:25,:-1]
y_train = train_nonlinear[:25,-1]
X_validate = train_nonlinear[25:,:]
y_validate = train_nonlinear[25:,-1]

print("Training set ..")
weights = linear_regression(X_train,y_train,X_validate,y_validate)

test_nonlinear = nonlinear_transform(test)
X_test = test_nonlinear[:,:-1]
y_test = test_nonlinear[:,-1]

print("Test set ..")
verify_regression(X_test,y_test,weights)
    
print("Reverse training set ..")
weights = linear_regression(X_validate,y_validate,X_train,y_train)

print("Reverse test set ..")
verify_regression(X_test,y_test,weights)

## Validation Bias
print("Validation bias problem ..")
e_out = []
for i in range(10000):
    e = np.random.uniform(0,1,size=(1,2))
    e_min = np.min(e)
    e_out.append(np.append(e,np.min(e)))
    
def split_train_validate(data_train, data_validate):
    X_validate = data_validate[0].reshape(-1,1)
    y_validate = data_validate[1].reshape(-1,1)
    X_train = data_train[:,0]
    y_train = data_train[:,1]
    return X_train, X_validate, y_train, y_validate

def calc_squared_error(actual, prediction):
    return sum((actual - prediction)**2)

def check_plot(X_train, y_train, X_validate, y_validate, x, predictions):
    plt.scatter(X_train,y_train,color="blue")
    plt.scatter(X_validate,y_validate,color="red")
    plt.plot(x, predictions)
    plt.show()

e_out = np.array(e_out)
e_out = np.mean(e_out,axis=0)
print("Expected values are: [{:.1f}, {:.1f}, {:.2f}]".format(e_out[0],e_out[1],e_out[2]))

## Cross Validation
print("Cross validation problem ..")
data = np.array([[-1,0],[1,0]])

errors_p = []
for p in range(10):
    data_p = np.vstack([data,[p,1]])
    
    errors_constant = []
    errors_linear = []
    for i in range(data_p.shape[0]):
        data_validate = data_p[i]
        data_train = data_p[np.arange(len(data_p))!=i]

        X_train, X_validate, y_train, y_validate = split_train_validate(data_train,data_validate)
        model_constant = np.polyfit(X_train,y_train,0)
        model_linear = np.polyfit(X_train,y_train,1)
        
        plot_constant = False
        if plot_constant:
            x = np.arange(np.min(data_p[:,0]),np.max(data_p[:,0])+1,1.)
            predictions = np.zeros(x.shape)
            predictions[...] = model_constant
            check_plot(X_train, y_train, X_validate, y_validate, x, predictions)
            
        plot_linear = False
        if plot_linear:
            x = np.arange(np.min(data_p[:,0]),np.max(data_p[:,0])+1,1.)
            predictions = x * model_linear[0] + model_linear[1]
            check_plot(X_train, y_train, X_validate, y_validate, x, predictions)
        
        constant_error = calc_squared_error(y_validate, model_constant)
        errors_constant.append(constant_error)
                
        linear_error = calc_squared_error(y_validate, X_validate*model_linear[0] + model_linear[1])
        errors_linear.append(linear_error)
        
    cv_error_constant = np.mean(errors_constant)
    cv_error_linear = np.mean(errors_linear)
    errors_p.append([p, cv_error_constant, cv_error_linear])

errors_p = np.array(errors_p)
errors_x = errors_p[:,0]
errors_y1 = errors_p[:,1]
errors_y2 = errors_p[:,2]
plt.plot(errors_x, errors_y1, label="constant")
plt.plot(errors_x, errors_y2, label="linear")
plt.legend()
plt.show()