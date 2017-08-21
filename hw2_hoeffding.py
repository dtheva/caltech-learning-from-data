import numpy as np

num_coins = 1000
num_flips = 10

num_iter = 100000

out_v_min = []
out_v_1 = []
out_v_rand = []

for i in range(num_iter):
    flips = np.random.randint(2,size=(num_coins,num_flips))
    num_heads = np.sum(flips,axis=1)
    
    c1 = 0
    cmin = np.argmin(num_heads)
    crand = np.random.randint(0,num_coins)
    
    out_v_1.append(num_heads[c1]/num_flips)
    out_v_min.append(num_heads[cmin]/num_flips)
    out_v_rand.append(num_heads[crand]/num_flips)
    
print("Average v1: %.2f" % np.mean(out_v_1))
print("Average v_rand: %.2f" % np.mean(out_v_rand))
print("Average v_min: %.2f" % np.mean(out_v_min))