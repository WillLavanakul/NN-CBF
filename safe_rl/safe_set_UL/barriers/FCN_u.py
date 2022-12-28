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
    self.FCN_a = ptu.build_mlp(input_size, 1, n_layers, size, activation)
    self.FCN_b = ptu.build_mlp(input_size, 1, n_layers, size, activation)
    self.loss_a = nn.MSELoss()
    self.loss_b = nn.MSELoss()
    self.optimizer_a = optim.Adam(self.FCN_a.parameters(),self.learning_rate)
    self.optimizer_b = optim.Adam(self.FCN_b.parameters(),self.learning_rate)

  def forward(self, x):
    x = ptu.from_numpy(x)
    return self.FCN_a(x), self.FCN_b(x)

  def update(self, x, u, u_bar, epoch):
    u = ptu.from_numpy(u)
    u_bar = ptu.from_numpy(u_bar)
    a, b = self.forward(x)
    if epoch > 90:
      u_bar = th.where(u_bar < 0, -1.0, 0)
    diff = -u_bar * (a*u-b)
    mask = th.where(diff > 0, 1, 0)

    self.optimizer_a.zero_grad()
    self.optimizer_b.zero_grad()
    loss = th.sum((diff*mask), dim=1).mean()
    loss.backward()
    self.optimizer_a.step()
    self.optimizer_b.step()
    return loss, loss

  def get_loss(self, x, u, u_bar):
    with th.no_grad():
      u = ptu.from_numpy(u)
      u_bar = ptu.from_numpy(u_bar)
      a, b = self.forward(x)
      diff = -u_bar * (a*u-b)
      mask = th.where(diff > 0, 1, 0)
      loss = th.sum((diff*mask)**2, dim=1).mean()
    return loss

  def get_hyp(self, x):
    a, b = self.forward(x)
    a, b = ptu.to_numpy(a), ptu.to_numpy(b)
    return a, b
