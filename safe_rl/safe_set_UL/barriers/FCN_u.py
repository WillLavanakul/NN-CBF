import infra.pytorch_utils as ptu
import numpy as np
import torch as th
from torch import nn
from torch import optim

class FCN_u():
  def __init__(self, input_size, output_size, n_layers, size, activation, lr):
    '''
    n_layers: number of hidden layers
    Size: dim of hidden layers
    '''
    self.learning_rate = lr
    self.FCN = ptu.build_mlp(input_size, output_size, n_layers, size, activation)
    self.loss = nn.MSELoss()
    self.optimizer = self.optimizer = optim.Adam(self.FCN.parameters(),
                                        self.learning_rate)

  def forward(self, x):
    x = ptu.from_numpy(x)
    return self.FCN(x)

  def update(self, x, u, u_bar):
    u = ptu.from_numpy(u)
    u_bar = ptu.from_numpy(u_bar)
    self.optimizer.zero_grad()
    pred = self.forward(x)
    a = pred[:, 0][:, None]
    b = pred[:, 1][:, None]
    u_hat = a*u-b
    loss = self.loss(u_hat, u_bar)
    loss.backward()
    self.optimizer.step()
    return loss

  def get_hyp(self, x):
    pred = ptu.to_numpy(self.forward(x))
    return pred[0], pred[1]