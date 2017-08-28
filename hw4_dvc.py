import numpy as np
import matplotlib.pyplot as plt

dvc=50
delta=.05

def orig_vc_bound(n):
#    print(n)
    return np.sqrt(8/n*np.log(4*((2*n)**dvc)/delta))

def rademacher(n):
    return np.sqrt(2*(dvc+1)*np.log(2*n)/n) + np.sqrt(2/n*np.log(1/delta)) + 1/n

def parrondo(e,n):
    return np.sqrt((2*e + np.log(6) + dvc*np.log(2*n) - np.log(delta))/n)

def devroye(e,n):
    return np.sqrt(((4*e*(1+e)) + dvc*np.log(n**2) + np.log(4/delta))/(2*n))

def orig_vc_bound2(n):
    return np.sqrt(8/n*np.log(4*2**(2*n)/delta))

def rademacher2(n):
    return np.sqrt(2*np.log(2*n*2**n)/n) + np.sqrt(2/n*np.log(1/delta)) + 1/n

def parrondo2(e,n):
    return np.sqrt((2*e + 2*n*np.log(2) + np.log(6/delta))/n)

def devroye2(e,n):
    return np.sqrt(((4*e*(1+e)) + 2*np.log(2**n) + np.log(4/delta))/(2*n))

def sum_bound(n,dvc):
    total = 0
    for i in range(0,dvc+1):
        total += np.math.factorial(n)/(np.math.factorial(i)*np.math.factorial(n-i))
    return total

N=1000

f1 = np.vectorize(orig_vc_bound)
f2 = np.vectorize(rademacher)
f3 = np.vectorize(parrondo)
f4 = np.vectorize(devroye)
f5 = np.vectorize(orig_vc_bound2)
f6 = np.vectorize(rademacher2)
f7 = np.vectorize(parrondo2)
f8 = np.vectorize(devroye2)
x = np.arange(1,N*2+1)
#
for e in np.arange(0,10,.2):
    print(e)
    plt.plot(x,f1(x),label="a")
    plt.plot(x,f2(x),label="b")
    plt.plot(x,f3(e,x),label="c")
    plt.plot(x,f4(e,x),label="d")

    plt.legend()
    plt.show()

#for e in np.arange(0,10,.2):
#    print(e)
#    plt.plot(x,f5(x),label="a")
#    plt.plot(x,f6(x),label="b")
#    plt.plot(x,f7(e,x),label="c")
#    plt.plot(x,f8(e,x),label="d")
#
#    plt.legend()
#    plt.show()
#    