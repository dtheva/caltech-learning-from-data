import numpy as np

eta = .1

def calc_error(u,v):
    return (u*np.e**v - 2*v*np.e**-u)**2

def calc_deriv_u(u,v):
    return 2 * (np.e**v + 2*v*np.e**-u) * (u*np.e**v - 2*v*np.e**-u)

def calc_deriv_v(u,v):
    return 2 * (u*np.e**v - 2*v*np.e**-u) * (u*np.e**v - 2*np.e**-u)

print("Calculating error for gradient descent ..")

w = np.array([1.,1.])
for i in range(21):
    u = w[0]
    v = w[1]
    error = calc_error(u,v)
    print("error: {}".format(error))
    if error < 10**-14:
        print("Error fell below 10^-14 at iteration: {}".format(i))
        break
    else:
        w += -eta * np.array([calc_deriv_u(u,v), calc_deriv_v(u,v)])
        
print("Final weights: [{},{}]".format(w[0],w[1]))

print("")
print("Calculating error for coordinate descent ..")

w = np.array([1.,1.])
for i in range(16):
    u = w[0]
    v = w[1]
    error = calc_error(u,v)
    print("error: {}".format(error))
    w += -eta * np.array([calc_deriv_u(u,v), 0])
    w += -eta * np.array([0, calc_deriv_v(w[0],w[1])]) #use update weights