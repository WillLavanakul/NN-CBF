import numpy as np

class inv_CBF():
  def __init__(self, c_a, a, b, m, g, l):
    self.c_a = c_a
    self.a = a
    self.b = b
    self.m = m
    self.g = g
    self.l = l

  def L_fh(self, x):
    theta = x[0]
    theta_dot = x[1]
    a, b, g, l = self.a, self.b, self.g, self.l
    return ((-2*theta*theta_dot)/(a**2))-((theta_dot**2)/(a*b))-np.sin(theta)*((2*theta_dot*g/(l*(b**2))) + theta*g/(l*a*b))

  def L_gh(self, x):
    theta = x[0]
    theta_dot = x[1]
    c_a, a, b, m, l = self.c_a, self.a, self.b, self.m, self.l
    return (-theta/(m*(l**2)*a*b)) - (2*theta_dot/(m*(l**2)*(b**2)))

  def neg_alpha(self, x):
    theta = x[0]
    theta_dot = x[1]
    c_a, a, b = self.c_a, self.a, self.b
    h = 0.3-((theta**2)/(a**2))-((theta_dot**2)/(b**2))-(theta*theta_dot/(a*b))
    return -c_a * h

  def H(self, x):
    A = np.array([
        [1/(self.a**2), 0.5/(self.a*self.b)],
        [0.5/(self.a*self.b), 1/(self.b**2)]
    ])
    return 0.3 - (x.T @ (A @ x))

  def get_hyp(self, x):
    return self.L_gh(x), self.neg_alpha(x) - self.L_fh(x)