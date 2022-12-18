from scipy.integrate import odeint
import numpy as np

class inv_ped():
  def __init__(self, g, m, l, u_min, u_max, delta_t):
    self.u_min = u_min
    self.u_max = u_max
    self.g = g
    self.m = m
    self.l = l
    self.delta_t = delta_t

  def run_system(self, x_0, total_time, input_signal):
    y = [x_0]
    x_t = x_0
    t = 0
    while t < total_time:
      timesteps = np.linspace(t, t+self.delta_t)
      u = input_signal.forward(x_t, t)
      y_delta_t = np.array(odeint(self.system, x_t, timesteps, args=(u,))[-1])
      y.append(y_delta_t)
      t += self.delta_t
      x_t = y_delta_t
    return np.array(y)

  def system(self, x, t, u):
    g = self.g
    m = self.m
    l = self.l
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
    