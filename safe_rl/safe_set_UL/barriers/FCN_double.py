import infra.pytorch_utils as ptu
import numpy as np
import torch as th
from torch import nn
from torch import optim

class FCN_double():
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

  def update(self, x, labels):
    a_labels = labels[:, 0]
    b_labels = labels[:, 1]
    a_labels = ptu.from_numpy(a_labels)
    b_labels = ptu.from_numpy(b_labels)
    self.optimizer_a.zero_grad()
    a, b = self.forward(x)
    loss_a = self.loss_a(a, a_labels)
    loss_a.backward()
    self.optimizer_a.step()
    self.optimizer_b.zero_grad()
    loss_b = self.loss_b(b, b_labels)
    loss_b.backward()
    self.optimizer_b.step()
    return loss_a, loss_b

  def update2(self, x, inputs, input_labels):
    inputs = th.tensor(inputs)
    a, b = self.forward(x)
    u_hat = a*th.tensor(inputs)-b
    u_bar = th.tensor(input_labels)
    #mask = th.where(u_bar < 0, 1.0, 0.0)
    diff = -u_bar*u_hat
    mask = th.where(diff > 0, 1, 0)
    # print("inputs", inputs)
    # print("input_labels", input_labels)
    # print("diff", diff)
    # print("mask", mask)

    self.optimizer_a.zero_grad()
    self.optimizer_b.zero_grad()
    loss = th.sum((diff*mask), dim=1).mean()
    loss.backward()
    self.optimizer_a.step()
    self.optimizer_b.step()
    
    return loss, loss


  def get_hyp(self, x):
    a, b = self.forward(x)
    a, b = ptu.to_numpy(a), ptu.to_numpy(b)
    return a, b