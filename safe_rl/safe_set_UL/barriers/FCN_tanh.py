import infra.pytorch_utils as ptu
import numpy as np
import torch as th
from torch import nn
from torch import optim

class FCN_tanh():
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
    return self.FCN_a(x).squeeze(), th.tanh(self.FCN_b(x)).squeeze()

  def update(self, x, labels):
    a_labels = labels[:, 0] / np.abs(labels[:, 0])
    b_labels = labels[:, 1] / np.abs(labels[:, 0])
    a_labels = ptu.from_numpy(a_labels)
    b_labels = ptu.from_numpy(b_labels)
    self.optimizer_a.zero_grad()
    a, b = self.forward(x)
    loss_a = self.loss_a(a, a_labels)
    loss_a.backward()
    self.optimizer_a.step()
    self.optimizer_b.zero_grad()
    loss_b = self.loss_b(th.atanh(b), b_labels)
    loss_b.backward()
    self.optimizer_b.step()
    return loss_a, loss_b

  def get_hyp(self, x):
    a, b = self.forward(x)
    a, b = np.sign(ptu.to_numpy(a)), np.arctanh(ptu.to_numpy(b))
    return a, b