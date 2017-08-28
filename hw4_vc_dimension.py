import math

def mh(i,q=1):
    if i == 1:
        return 2
    else:
        return 2*mh(i-1,q) - nCr(i-1,q)

def nCr(n,r):
    if r > n:
        return 0
    f = math.factorial
    return f(n) / f(r) / f(n-r)    

def calc_dvc(q=1):
    for n in range(1,150):
        if mh(n,q) != 2**n:
            return n-1
        
for q in range(1,11):
    print("q: {}, dvc: {}".format(q,calc_dvc(q)))