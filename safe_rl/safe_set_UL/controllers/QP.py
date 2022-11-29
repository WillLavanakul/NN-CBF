import cvxpy as cp
import numpy as np

class QP_CBF_controller():
  def __init__(self, target, barrier):
    self.target = target
    self.cbf = barrier
    self.u_min = -3.0
    self.u_max = 3.0

  def forward(self, x, t):
    a,b = self.cbf.get_hyp(x)
    target_u = self.target(x, t)
    u = cp.Variable()
    #print("x", x, "inv", np.sign(self.H(x)), "plane", (a, b), "t", t)
    prob = cp.Problem(cp.Minimize((1/2)*((u-target_u)**2)),
                      [a*u >= b]
                    )
    prob.solve()
    #print("target u and solved u", target_u, u.value)
    if u.value < 0:
      return max(u.value, self.u_min)
    else:
      return min(u.value, self.u_max) 

  def H(self, x):
    A = np.array([
        [1/(0.075**2), 0.5/(0.075*0.15)],
        [0.5/(0.075*0.15), 1/(0.15**2)]
    ])
    return 0.3 - (x.T @ (A @ x))