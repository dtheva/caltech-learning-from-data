import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(X,y,title=None):
    pos = X[y==1]
    neg = X[y==-1]
    
    plt.title(title)
    plt.scatter(pos[:,0],pos[:,1],color="blue")
    plt.scatter(neg[:,0],neg[:,1],color="red")
    plt.show()
    
#calculate nonlinear features for feature set with dimension 2
def calc_nonlinear(X):
    x1 = np.ones(X.shape[0])
    x2 = X
    x3 = X**2
    x4 = X[:,0] * X[:,1]
    x5 = np.abs(X[:,0] - X[:,1])
    x6 = np.abs(X[:,0] + X[:,1])
    return np.hstack([x1.reshape(-1,1),x2,x3,x4.reshape(-1,1),x5.reshape(-1,1),x6.reshape(-1,1)])

def calc_linear_weights(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def calc_linear_weights_regularized(X,y,lam):
    w = X.T.dot(X)
    w += (lam*np.identity(w.shape[0]))
    w = np.linalg.inv(w).dot(X.T).dot(y)
    return w

def get_lambda(k):
    return 10**k

train = np.fromfile("in.dta",sep=" ").reshape(-1,3)
test = np.fromfile("out.dta",sep=" ").reshape(-1,3)

X_train = train[:,:-1]
y_train = train[:,-1]
X_test = test[:,:-1]
y_test = test[:,-1]

plot_scatter(X_train,y_train,title="train data")

w = calc_linear_weights(X_train,y_train)
prediction = np.sign(X_train.dot(w))
plot_scatter(X_train,prediction,title="initial regression")

X_train_nonlinear = calc_nonlinear(X_train)
w = calc_linear_weights(X_train_nonlinear,y_train)
prediction = np.sign(X_train_nonlinear.dot(w))
plot_scatter(X_train,prediction,title="nonlinear regression")

X_test_nonlinear = calc_nonlinear(X_test)
prediction_test = np.sign(X_test_nonlinear.dot(w))
plot_scatter(X_test,prediction_test,title="nonlinear regression (oos)")

error_insample = len(prediction[prediction != y_train])/len(y_train)
error_outsample = len(prediction_test[prediction_test != y_test])/len(y_test)
print("in-sample error: %.2f" % error_insample)
print("out-sample error: %.2f" % error_outsample)

errors = []
for k in [2,1,0,-1,-2]:
    lam = get_lambda(k=k)
    w_reg = calc_linear_weights_regularized(X_train_nonlinear,y_train,lam)
    prediction_reg = np.sign(X_train_nonlinear.dot(w_reg))
    plot_scatter(X_train,prediction_reg,title="nonlinear regression with regularization")
    
    prediction_reg_test = np.sign(X_test_nonlinear.dot(w_reg))
    plot_scatter(X_test,prediction_reg_test,title="nonlinear regression with regularization (oos)")
    
    error_reg_insample = len(prediction_reg[prediction_reg != y_train])/len(y_train)
    error_reg_outsample = len(prediction_reg_test[prediction_reg_test != y_test])/len(y_test)
    print("in-sample error (reg,k=%d): %.2f" % (k, error_reg_insample))
    print("out-sample error (reg,k=%d): %.2f" % (k, error_reg_outsample))

    errors.append([k,error_reg_outsample])
    

plt.plot([x[0] for x in errors],[x[1] for x in errors])
plt.title("Error by k")
plt.show()