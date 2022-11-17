import infra.pytorch_utils as ptu
import numpy as np
import torch as th
from torch import nn
from torch import optim

class FCN():
  def __init__(self, input_size, output_size, n_layers, size, activation, lr):
    '''
    n_layers: number of hidden layers
    Size: dim of hidden layers
    '''
    self.learning_rate = lr
    self.FCN = ptu.build_mlp(input_size, output_size, n_layers, size, activation)
    self.loss = nn.MSELoss()
    self.optimizer = optim.Adam(self.FCN.parameters(),self.learning_rate)

  def forward(self, x):
    x = ptu.from_numpy(x)
    return self.FCN(x)

  def update(self, x, labels):
    labels = ptu.from_numpy(labels)
    pred = self.forward(x)
    self.optimizer.zero_grad()
    loss = self.loss(pred, labels)
    loss.backward()
    self.optimizer.step()
    return loss, loss

  def get_hyp(self, x):
    pred = ptu.to_numpy(self.forward(x))
    return pred[0], pred[1]