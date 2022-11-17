from scipy.integrate import odeint
import numpy as np

class inv_ped():
  def __init__(self, g, m, l, u_min, u_max):
    self.u_min = u_min
    self.u_max = u_max
    self.g = g
    self.m = m
    self.l = l

  def run_system(self, x_0, timesteps, input_signal):
    return odeint(self.system, x_0, timesteps, args=(input_signal, ))

  def system(self, x, t, controller):
    g = self.g
    m = self.m
    l = self.l
    u = controller.forward(x, t)
    theta = x[0]
    theta_dot = x[1]
    x_dot = [theta_dot, (g/l)*np.sin(x[0]) +  u/(m*l**2)]
    return x_dot

  def next_state(self, x, u, delta_t):
    g = self.g
    m = self.m
    l = self.l
    theta = x[0]
    theta_dot = x[1]
    return x + delta_t * np.array([theta_dot, (g/l)*np.sin(x[0]) +  u/(m*l**2)])
    