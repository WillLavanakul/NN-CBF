from infra import utils
from barriers.disc import disc
import matplotlib.pyplot as plt
import numpy as np
from envs.inv_ped import inv_ped
from barriers.inv_ped_CBF import inv_CBF

n = 100
x1 = np.linspace(-0.05, 0.05, n)
x2 = np.linspace(-0.1, 0.1, n)
z_ba = -3.0*np.ones((n, n))
z_a = -3.0*np.ones((n, n))
z_b = -3.0*np.ones((n, n))

c_a, a, b, m, g, l = 0.2, 0.075, 0.15, 2, 10, 1
delta_t = 0.05
env = inv_ped(g, m, l, -3, 3, delta_t)
cbf1 = inv_CBF(c_a, a, b, m, g, l)

for i in range(n):
    for j in range(n):
        x = np.array([[x1[i], x2[j]]])
        inputs = utils.sample_inputs(x, 1000)
        input_labels = utils.label_inputs(x, inputs, env, cbf1, delta_t)
        label = utils.label_states(x, inputs, input_labels)[0]
        z_ba[i, j] = label[1] / label[0]
        label = label / np.linalg.norm(label)
        z_a[i, j] = label[0]
        z_b[i, j] = label[1]


fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.title.set_text("b/a")

X, Y = np.meshgrid(x1, x2)
ax.plot_surface(X, Y, z_ba)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.title.set_text("a")
ax.plot_surface(X, Y, z_a)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.title.set_text("b")
ax.plot_surface(X, Y, z_b)

plt.savefig('ab_norm.png'.format(delta_t))