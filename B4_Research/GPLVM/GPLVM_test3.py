import numpy as np
import GPy.models
from scipy.io import loadmat
import scipy.io as spio
import GPy
import matplotlib.pyplot as plt

data = np.loadtxt('right_hand_swing.txt')
# print(data)
# print(np.shape(data))
input_dim = 2

kernel = GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Bias(input_dim) + GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim)
model = GPy.models.GPLVM(Y=data, input_dim=input_dim)
model.optimize(messages=True)
model.plot_latent()
plt.savefig('fig_right_hand_swing2.png')
