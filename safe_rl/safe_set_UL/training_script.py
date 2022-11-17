from barriers.FCN import FCN
from barriers.FCN_double import FCN_double
from barriers.FCN_tanh import FCN_tanh
from barriers.inv_ped_CBF import inv_CBF
from controllers.QP import QP_CBF_controller
from envs.inv_ped import inv_ped
from infra import utils
import matplotlib.pyplot as plt
import numpy as np

model = FCN(2, 2, 4, 1000, 'relu', 5e-3)

c_a, a, b, m, g, l = 0.2, 0.075, 0.15, 2, 10, 1
u_min, u_max, x_min, x_max = -3, 3, [-0.3, -0.6], [0.3, 0.6]
delta_t = 0.1
cbf1 = inv_CBF(c_a, a, b, m, g, l)
env = inv_ped(g, m, l, u_min, u_max)
normalized = False

print('Collecting data...')
normalize = False
data, labels, all_inputs, all_input_labels = utils.get_data(1000, 200, delta_t, env, cbf1)

print("num data", len(data))
print("states", data[:3])
print('labels', labels[:3])
train_size = 1000
val_size = 100
epochs = 50
batch_size = 32
train_data, train_labels = data[:train_size], labels[:train_size]
val_data, val_labels = data[:val_size], labels[:val_size]

print("Starting training...")
unnorm_losses_a = []
unnorm_losses_b = []
model.FCN.train()
for epoch in range(epochs):
  print("Epoch:", epoch)
  s_train_data, s_train_labels = utils.shuffle_data(train_data, train_labels)
  running_loss_a = 0
  running_loss_b = 0
  for batch in range(train_size // batch_size):
    batch_data = s_train_data[batch*batch_size:batch*batch_size+batch_size]
    batch_labels = s_train_labels[batch*batch_size:batch*batch_size+batch_size]
    batch_loss_a, batch_loss_b = model.update(batch_data, batch_labels)
    running_loss_a += batch_loss_a
    running_loss_b += batch_loss_b
  print("Train loss:", running_loss_a, running_loss_b)
  unnorm_losses_a.append(running_loss_a.item())
  unnorm_losses_b.append(running_loss_b.item())

plt.plot(range(len(unnorm_losses_a)), unnorm_losses_a, label='train loss a')
plt.plot(range(len(unnorm_losses_b)), unnorm_losses_b, label='train loss b')
plt.legend()
plt.title("training loss")
plt.ylabel('loss (MSE)')
plt.xlabel('epoch')
plt.savefig("training_plots/training_curve_dt{0}.png".format(str(delta_t)))
plt.clf()

# Post training stuff
def target_input(x, t):
  # target input for jason paper
  if t < 2:
    return 3
  elif t < 4:
    return -3
  elif t < 6:
    return 3
  else:
    return m*(l**2)*((-g/l)*np.sin(x[0]) - (1.5*x[0]+1.5*x[1]))

NN_controller = QP_CBF_controller(target_input, model)
cbf_controller = QP_CBF_controller(target_input, cbf1)

# input plots
print("Creating input plots...")
n = 50
x1 = np.linspace(-0.05, 0.05, n)
x2 = np.linspace(-0.1, 0.1, n)
z_cbf = np.zeros((n, n))
z_nn = np.zeros((n, n))
for i in range(n):
  for j in range(n):
    x = np.array([x1[i], x2[j]])
    if cbf1.H(x) >= 0:
      u_cbf = cbf_controller.forward(x, 0)
      u_nn = NN_controller.forward(x, 0)
      z_cbf[i, j] = u_cbf
      z_nn[i, j] = u_nn
 
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')

X, Y = np.meshgrid(x1, x2)
ax.plot_surface(X, Y, z_cbf)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, z_nn)


ax.axes.set_xlim3d(left=-0.05, right=0.05) 
ax.axes.set_ylim3d(bottom=-0.1, top=0.1) 
ax.axes.set_zlim3d(bottom=-3, top=3) 

plt.savefig('input_plots/input_dt{0}.png'.format(str(delta_t)))
plt.clf()

# eval on control
print("Running system...")

system = inv_ped(10, 2, 1, -3, 3)
x = [0.01, -0.05]
t = np.linspace(0, 10, 100)
y_NN = system.run_system(x, t, NN_controller)
y_CBF = system.run_system(x, t, cbf_controller)
plt.xlim(-0.3, 0.3)
plt.ylim(-0.6, 0.6)
plt.plot(y_NN[:,0], y_NN[:, 1], label="NN")
plt.plot(y_CBF[:,0], y_CBF[:, 1], label="CBF")
plt.xlabel('theta')
plt.ylabel('theta_dot')
plt.title('CBF vs NN')
plt.legend()
plt.savefig("controller_plots/controller_dt{0}.png".format(str(delta_t)))