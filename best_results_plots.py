import matplotlib.pyplot as plt 
import numpy as np

# data to work with 
# d = [r2 train, rmse train, r2 test, rmse train]

d1 = [0.99994415,	0.634763956,    0.998814642,    1.080364227]
d2 = [0.999976933,	0.479294688,	0.995796978,	1.6356529]
d3 = [0.999921799,	0.615314901,	0.997834444,	1.187758207]
d4 = [0.999856949,	0.688983738,	0.997632921,	1.255249977]
d5 = [0.999927461,	0.543871105,	0.999167383,	1.005594254]
d6 = [0.999965906,	0.468604386,	0.998050392,	1.035091043]
d7 = [0.999924004,	0.604539096, 	0.997920394, 	1.139514804]
d8 = [0.999894559,	0.601138175,	0.997507453,	1.270525336]
d9 = [0.999949098,	0.530707538,    0.997727752, 	1.443289757]
d10 = [0.999973476,	0.504232883, 	0.999004662, 	1.210773945]
d11 = [0.999914348,	0.657282114,	0.998991013,	0.960251153]
d12 = [0.999941587,	0.556033611,	0.999120474,	1.015586138]
d13 = [0.999894321,	0.648015916,	0.999352098,	1.056265593]
d14 = [0.999958217,	0.538418531,	0.998433888,	1.23666966]
d15 = [0.999476552,	0.98650533,	    0.995934069,	1.478081465]
d16 = [0.999943018,	0.528552294,	0.997718513,	1.306635737]
d17 = [0.999951959,	0.551995456,	0.9995839,	    0.957433343]
d18 = [0.999966204,	0.453334898,	0.998744071,	1.049260139]
d19 = [0.99997282,	0.438910931,	0.998937368,	1.311888218]

datasets = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19]

def r2_train() : # function to return value for r^2 (train) from dataset 
    r2_train_array = []

    for d in datasets : 
        r2_train_array.append(d[0])

    return r2_train_array 

def r2_test() : # function to return value for r^2 (test) from dataset 
    r2_test_array = []
     
    for d in datasets : 
        r2_test_array.append(d[2])

    return r2_test_array 

def rmse_train() : # function to return value for rmse (train) from dataset  
    rmse_train_array = []
     
    for d in datasets : 
        rmse_train_array.append(d[1])

    return rmse_train_array

def rmse_test() : # function to return value for rmse (test) from dataset 
    rmse_test_array = []
     
    for d in datasets : 
        rmse_test_array.append(d[3])

    return rmse_test_array

r2_train_vals = r2_train()
r2_test_vals = r2_test()
rmse_train_vals = rmse_train()
rmse_test_vals = rmse_test()

# plt.scatter(x, rmse_train_vals, label='RMSE Train', marker='o')
# plt.scatter(x, rmse_test_vals, label='RMSE Test', marker='o')

# plt.title("RMSE VALUES")
# plt.xticks(x)
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.scatter(r2_train_vals, r2_test_vals, marker = "o" )

# for i in range(len(r2_train_vals)):
#     plt.text(r2_train_vals[i], r2_test_vals[i], str(i+1), fontsize=9, ha='right', va='bottom')

# plt.title("R^2 Comparisons")
# plt.xlim(0.9994, 1) 
# plt.xlabel("r^2 (train)")
# plt.ylabel("r^2 (test)")
# plt.show()

min_val = min(min(rmse_train_vals), min(rmse_test_vals))
max_val = max(max(rmse_train_vals), max(rmse_test_vals))

x = np.linspace(min_val, max_val, 100)
y = x

plt.scatter(rmse_train_vals, rmse_test_vals, marker = "o")
plt.plot(x,y)

for i in range(len(rmse_train_vals)):
    plt.text(rmse_train_vals[i], rmse_test_vals[i], str(i+1), fontsize=9, ha='right', va='bottom')

plt.title("RMSE Comparisons")
plt.xlabel("rmse (train)")
plt.ylabel("rmse (test)")
plt.show()