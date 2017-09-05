import numpy as np
from hw1_perceptron import PLA
import matplotlib.pyplot as plt
#import quadprog #revisit at another time
from sklearn.svm import SVC

d=2
N=100
num_iter = 1000

def generate_data(n,d):
    return np.random.uniform(-1,1,(n,d))

def create_target_function():
    X = generate_data(2,d)
    target_function = np.polyfit(X[:,0],X[:,1],1) # only works for d=2
    
    return target_function

def plot_line(func):
    X = np.array([-1,1])
    plt.plot(X,X*func[0] + func[1])

def plot_scatter(data,color="red"):
    plt.scatter(data[:,0],data[:,1],color=color)
    
def show_plot():
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xticks(())
    plt.yticks(())
    plt.show()

def calculate_y(X, func):
    x0 = X[:,0]
    y0 = X[:,1]
    slope = func[0]
    intercept = func[1]
    y_target = np.zeros(x0.shape)
    y_target[x0 * slope + intercept > y0] = 1
    y_target[x0 * slope + intercept < y0] = -1
    return y_target

def create_dataset():
    target_function = create_target_function()
    data = generate_data(N,d)
    y = calculate_y(data, target_function)
    return target_function, data, y

def plot_data(target_function, X, y):
    plot_line(target_function)
    plot_scatter(X[y==1],"red")
    plot_scatter(X[y==-1],"blue")
    show_plot()
    
def plot_svm(model):
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-1, 1)
    yy = a * xx - (model.intercept_[0]) / w[1]
    
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    
    plt.plot(xx, yy, "k-")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy_up, "k--")
    
    plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],s=80, facecolors="none", zorder=10, edgecolors="k")
    plt.scatter(data[:,0],data[:,1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolors="k")
    plt.axis("tight")
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
    
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    show_plot()

def calc_error(y_predict,y_test):
    return len(y_predict[y_predict!=y_test])/len(y_predict)

errors_pla = []
errors_svm = []
num_svs = []
for i in range(num_iter):
    target_function, data, y = create_dataset()
    while (sum(y) == N or sum(y) == -N):
        target_function, data, y = create_dataset()
    
    check_data = False
    if check_data:
        plot_data(target_function, data, y)
        
    pla = PLA()
    pla.train(data,y)
    
    num_test = 10000
    X_test = generate_data(num_test,d)
    y_test = calculate_y(X_test,target_function)
        
    y_predict = pla.predict(X_test)
    
    error_pla = calc_error(y_predict, y_test)
    errors_pla.append(error_pla)
    
    check_prediction = False
    if check_prediction:
        plot_data(target_function,X_test,y_predict)
         
    model = SVC(C=200000,kernel="linear")
    model.fit(data,y)
    num_svs.append(len(model.support_))
    
    check_svm = False
    if check_svm:
        plot_svm(model)
        
    y_predict = model.predict(X_test)
    error_svm = calc_error(y_predict, y_test)
    errors_svm.append(error_svm)
#    break
    
# Revisit quadratic programming at another time
#    P = np.outer(y,y) * np.dot(data,data.T)
##    q = (np.ones(data.shape[0])*-1)
#    q = -1 * np.ones((data.shape[0],1))
#    A = y
#    b = np.zeros(1)
##    G = np.vstack([np.eye(data.shape[0]) * -1, np.eye(data.shape[0])])
#    G = -1 * np.identity(data.shape[0])
##    h = np.hstack([np.zeros(data.shape[0]),np.ones(data.shape[0])*np.inf])
#    h = np.zeros((data.shape[0],1))
#    
#    qp_a = -q
#    qp_C = -np.vstack([A,G]).T
#    qp_b = -np.hstack([b,h])
#    meq = A.shape[0]
#    result = quadprog.solve_qp(P,qp_a,qp_C,qp_b,meq)
#    break

errors_pla = np.array(errors_pla)
errors_svm = np.array(errors_svm)
print("Error (PLA): %.2f" % np.mean(errors_pla))
print("Error (SVM): %.2f" % np.mean(errors_svm))
print("SVM performs better: [ {} / {} ]".format(sum(errors_svm < errors_pla),num_iter))
print("Average number of support vectors: {0:.0f}".format(np.mean(num_svs)))