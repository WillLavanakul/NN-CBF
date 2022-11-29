import cvxpy as cp
import numpy as np

class u_controller():
  def __init__(self, u):
    self.u = u

  def forward(self, x, t):
    return self.u