"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import deepxde as dde
import numpy as np
import torch
# print(torch.cuda.is_available())

# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)


device = torch.device("cpu")

dde.config.real.set_float64()

def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    # x = torch.from_numpy(x)
    return (x * np.sin(5 * x))


geom = dde.geometry.Interval(-1, 1)
num_train = 16
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)



activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [20] * 3 + [1], activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)