import pandas as pd, numpy as np
from hw6_regularization import calc_linear_weights_regularized
from hw8 import filter_one_vs_one
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def filter_one_vs_all(df,x1):
    df = df.copy()
    df["target"] = -1
    df.loc[df.digit == x1, "target"] = 1
    return df

def calc_binary_error(y1,y2):
    return len(y1[y1!=y2])/len(y1)

def feature_transform(X):
    z1 = (X[:,1] * X[:,2]).reshape(-1,1)
    z2 = X[:,1:]**2
    return np.hstack([X,z1,z2])

def get_X(df):
    return np.hstack([np.ones(df.shape[0]).reshape(-1,1),df[["intensity","symmetry"]].values])

def predict_linear(X,w):
    return np.sign(X.dot(w))

def get_data():
    data = np.array([
                [1, 0, -1],
                [0, 1, -1],
                [0, -1, -1],
                [-1, 0, 1],
                [0, 2, 1],
                [0, -2, 1],
                [-2, 0, 1]
            ])
    return data[:,:-1],data[:,-1]

def feature_transform2(X):
    x1 = X[:,0]
    x2 = X[:,1]
    z1 = (x2**2 - 2*x1 - 1).reshape(-1,1)
    z2 = (x1**2 - 2*x2 + 1).reshape(-1,1)
    return np.hstack([z1,z2])

def plot_scatter(X,y):
    plt.scatter(X[:,0],X[:,1],c=y,cmap="cool")
    plt.show()

def p7_9():
    results = []
    for x in range(0,10):
        df_one = filter_one_vs_all(df_train,x)
        df_one_test = filter_one_vs_all(df_test,x)
        
        X = get_X(df_one)
        y = df_one.target.values
        w = calc_linear_weights_regularized(X,y,1)
        
        X_test = get_X(df_one_test)
        y_test = df_one_test.target.values
        
        X_transformed = feature_transform(X)
        X_test_transformed = feature_transform(X_test)
        
        w_transformed = calc_linear_weights_regularized(X_transformed,y,1)
    
        y_predict = predict_linear(X,w)
        y_transformed = predict_linear(X_transformed,w_transformed)
        
        y_test_predict = predict_linear(X_test,w)
        y_test_transformed = predict_linear(X_test_transformed, w_transformed)
        
        e_in = calc_binary_error(y,y_predict)
        e_in_transform = calc_binary_error(y,y_transformed)
        e_out = calc_binary_error(y_test,y_test_predict)
        e_out_transform = calc_binary_error(y_test,y_test_transformed)
        
        results.append([x, e_in, e_in_transform, e_out, e_out_transform])
        
        print("error_in ({} vs all): {:.3f}".format(x,e_in))
        print("error_out ({} vs all) (feature transformed): {:.3f}".format(
                x,e_out_transform))
        
    results = pd.DataFrame(results,columns=["target","e_in","e_in_transformed", "e_out", "e_out_transformed"])
    results.plot(x="target",figsize=(13,9),title="Target vs. All")
    plt.show()


def p10():
    results = []
    for lam in [.01,.5,1]:
        x1 = 1
        x2 = 5
        df_tmp = filter_one_vs_one(df_train,x1,x2)
        df_tmp_test = filter_one_vs_one(df_test,x1,x2)
        
        X = get_X(df_tmp)
        y = df_tmp.digit.values
        X_test = get_X(df_tmp_test)
        y_test = df_tmp_test.digit.values
        
        X_transformed = feature_transform(X)
        X_test_transformed = feature_transform(X_test) 
        w_transformed = calc_linear_weights_regularized(X_transformed, y, lam)
        
        e_in = calc_binary_error(y,predict_linear(X_transformed, w_transformed))
        e_out = calc_binary_error(y_test,predict_linear(X_test_transformed, w_transformed))
        
        results.append([lam, e_in, e_out])
    
    results = pd.DataFrame(results,columns=["lambda","e_in","e_out"])
    results.plot(x="lambda",title="1 vs 5 (Feature Transformed)",style="o-",figsize=(13,9))
    plt.show()

def p11_12():
    X, y = get_data()
    plot_scatter(X,y)
    
    z = feature_transform2(X)
    plot_scatter(z,y)
    
    weights = [
                [-1,1,-.5],
                [1,-1,-.5],
                [1,0,-.5],
                [0,1,-.5]
            ]
    
    for label, weight in zip(["a","b","c","d",],weights):
        w = weight[:-1]
        b = weight[-1]
        out = z.dot(w) + b
        print("prediction ({}):".format(label), np.sign(out))
        print("actual:", y)
        
    svc = SVC(C=2**15,kernel="poly",degree=2,gamma=1,coef0=1)
    svc.fit(X,y)
    print("# of Support Vectors: {}".format(len(svc.support_vectors_)))
          
def generate_data(size=100):
    X = np.random.uniform(-1,1,(size,2))
    y = np.sign(X[:,1] - X[:,0] + .25*np.sin(np.pi*X[:,0]))
    return X, y

def calc_theta(X, centers, gamma):
    theta = np.empty((X.shape[0],centers.shape[0]+1))
    
    for k in range(centers.shape[0]):
        norm = np.linalg.norm(X - centers[k],axis=1)
        theta[:,k] = np.exp(-gamma*norm**2)
    
    theta[:,-1] = 1
    return theta 

def calc_rbf_weights(X, y, centers, gamma):
    theta = calc_theta(X, centers, gamma)
    w = np.linalg.pinv(theta.T.dot(theta)).dot(theta.T).dot(y)
#    w = np.linalg.lstsq(theta.T.dot(theta),theta.T.dot(y))[0]
    
    return w

def calc_rbf_predict(w, X, centers, gamma):
    theta = calc_theta(X, centers, gamma)
    return np.sign(theta.dot(w))

def compare_regular_vs_kernel(K=9,gamma=1.5,num_iter=100):
    results = []
    for i in range(num_iter):
        X, y = generate_data()
        lloyds = KMeans(n_clusters=K,init="random")
        lloyds.fit(X,y)
        
        while len(np.unique(lloyds.labels_)) != K:
            X, y = generate_data()
            lloyds.fit(X,y)
            
        svc = SVC(C=2**15,gamma=gamma)
        svc.fit(X,y)
        predict = svc.predict(X)
        error_in = calc_binary_error(y, predict)
        
        X_test, y_test = generate_data(1000)
        error_out = calc_binary_error(y_test,svc.predict(X_test))
        
        centers = lloyds.cluster_centers_  
        rbf_weights = calc_rbf_weights(X, y, centers, gamma)
        predict_rbf = calc_rbf_predict(rbf_weights, X, centers, gamma)
        predict_rbf_test = calc_rbf_predict(rbf_weights, X_test, centers, gamma)
        error_rbf_in = calc_binary_error(y, predict_rbf)
        error_rbf_out = calc_binary_error(y_test, predict_rbf_test)
        
        results.append([error_in,error_out,error_rbf_in,error_rbf_out])
    
    results = np.array(results)
    results_ein = results[:,0]
    results_eout = results[:,1]
#    results_ein_rbf = results[:,2]
    results_eout_rbf = results[:,3]
    
    print("# of times RBF failed to converge: [ {} / {} ]".format(len(results_ein[results_ein!=0]),len(results_ein)))
    print("# of times kernel beat regular version(eout): [ {} / {} ]".format(len(results_eout[results_eout < results_eout_rbf]),len(results_eout)))
    return results

def compare_ein_eout(results1,results2):  
    print("Ein goes down but Eout goes up %d times." % len(
            results2[(results2[:,2] < results1[:,2])
            &(results2[:,3] > results1[:,3])]))
    print("Ein goes up but Eout goes down %d times." % len(
            results2[(results2[:,2] > results1[:,2])
            &(results2[:,3] < results1[:,3])]))        
    print("Ein and Eout go up %d times." % len(
            results2[(results2[:,2] > results1[:,2])
            &(results2[:,3] > results1[:,3])]))
    print("Ein and Eout go down %d times." % len(
            results2[(results2[:,2] < results1[:,2])
            &(results2[:,3] < results1[:,3])]))
    print("Ein and Eout remain the same %d times." % len(
            results2[(results2[:,2] == results1[:,2])
            &(results2[:,3] == results1[:,3])]))
            
def p13_14():
    compare_regular_vs_kernel(K=9)
    
def p15():
    compare_regular_vs_kernel(K=12)

def p16():
    results_k9 = compare_regular_vs_kernel(K=9)
    results_k12 = compare_regular_vs_kernel(K=12)
    compare_ein_eout(results_k9,results_k12)
     
def p17():
    results_gamma1 = compare_regular_vs_kernel()
    results_gamma2 = compare_regular_vs_kernel(gamma=2)
    compare_ein_eout(results_gamma1, results_gamma2)
    
def p18():
    results = compare_regular_vs_kernel(K=9,gamma=1.5)
    print("Regular RBF achieves Ein=0 [ {} / {} ] times.".format(
            len(results[results[:,3]==0]),len(results)))
    
df_train = pd.read_csv("features2.train",delim_whitespace=True,header=None)
df_test = pd.read_csv("features2.test",delim_whitespace=True,header=None)

cols = ["digit", "intensity", "symmetry"]

df_train.columns = cols
df_test.columns = cols

#Regularized Linear Regression
p7_9()
p10()
#SVM
p11_12()
  
#RBF
p13_14()
p15()
p16()
p17()
p18()