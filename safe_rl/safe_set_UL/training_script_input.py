from barriers.FCN_u import FCN_u
from barriers.FCN_tanh import FCN_tanh
from barriers.inv_ped_CBF import inv_CBF
from controllers.QP import QP_CBF_controller
from envs.inv_ped import inv_ped
from infra import utils
import matplotlib.pyplot as plt
from barriers.disc import disc
import numpy as np

model = FCN_u(2, 2, 6, 300, 'relu', 5e-3)

c_a, a, b, m, g, l = 0.2, 0.075, 0.15, 2, 10, 1
u_min, u_max, x_min, x_max = -3, 3, [-0.3, -0.6], [0.3, 0.6]
delta_t = 0.05
cbf1 = inv_CBF(c_a, a, b, m, g, l)
env = inv_ped(g, m, l, u_min, u_max)
train_size = 1000
val_size = 1000
num_inputs = 200
epochs = 5
batch_size = 32

print('Collecting data...')
data, labels, all_inputs, all_input_labels = utils.get_data(train_size + val_size, num_inputs, delta_t, env, cbf1)
print(data.shape, labels.shape, all_inputs.shape, all_input_labels.shape)

print("num data", len(data))
print("states", data[:3])
print('labels', labels[:3])
train_data, inputs, input_labels = data[:train_size], all_inputs[:train_size], all_input_labels[:train_size]
val_data, val_inputs, val_input_labels, = data[train_size:], all_inputs[train_size:], all_input_labels[train_size:]

def hyperplane_vis():
  figure, axis = plt.subplots(10, 2)
  for i in range(10):
    x_v = val_data[i]
    a_v, b_v = model.get_hyp(x_v)
    axis[i, 0].scatter(val_inputs[i], val_input_labels[i])
    axis[i, 0].scatter(b_v/a_v, 0)

    x_t = train_data[i]
    a_t, b_t = model.get_hyp(x_t)
    axis[i, 1].scatter(inputs[i], input_labels[i])
    axis[i, 1].scatter(b_t/a_t, 0)
  plt.savefig("hyperplanes/hyperplane{0}.png".format(epochs))

  plt.clf()

print("Starting training...")
train_losses = []
val_losses = []
model.FCN_a.train()
model.FCN_b.train()
for epoch in range(epochs):
  print("Epoch:", epoch)
  s_train_data, s_inputs, s_input_labels = utils.shuffle_data_u(train_data, inputs, input_labels)
  running_loss = 0
  for batch in range(train_size // batch_size):
    batch_data = s_train_data[batch*batch_size:batch*batch_size+batch_size]
    batch_inputs = s_inputs[batch*batch_size:batch*batch_size+batch_size]
    batch_input_labels = s_input_labels[batch*batch_size:batch*batch_size+batch_size]
    batch_loss = model.update(batch_data, batch_inputs, batch_input_labels)
    running_loss += batch_loss
  val_loss = model.get_loss(val_data, val_inputs, val_input_labels)
  print("Val loss:", val_loss)
  val_losses.append(val_loss.item())
  running_loss = model.get_loss(train_data, inputs, input_labels)
  print("Train loss:", running_loss)
  train_losses.append(running_loss.item())

hyperplane_vis()

plt.plot(range(len(val_losses)), val_losses, label='val losses')
plt.plot(range(len(train_losses)), train_losses, label='train loss')
plt.legend()
plt.title("training loss")
plt.ylabel('loss (MSE)')
plt.xlabel('epoch')
plt.savefig("training_plots/uloss_training_curve_dt{0}.png".format(str(delta_t)))
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

disc1 = disc(cbf1, env, delta_t, 200)
NN_controller = QP_CBF_controller(target_input, model)
cbf_controller = QP_CBF_controller(target_input, cbf1)
disc_controller = QP_CBF_controller(target_input, disc1)

# input plots
print("Creating input plots...")
n = 50
x1 = np.linspace(-0.05, 0.05, n)
x2 = np.linspace(-0.1, 0.1, n)
z_cbf = -3.0*np.ones((n, n))
z_nn = -3.0*np.ones((n, n))
z_disc = -3.0*np.ones((n, n))
for i in range(n):
  for j in range(n):
    x = np.array([x1[i], x2[j]])
    if cbf1.H(x) >= 0:
      u_cbf = cbf_controller.forward(x, 0)
      u_nn = NN_controller.forward(x, 0)
      u_disc = disc_controller.forward(x, 0)
      z_cbf[i, j] = u_cbf
      z_nn[i, j] = u_nn
      z_disc[i, j] = u_disc
fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.title.set_text("cbf")

X, Y = np.meshgrid(x1, x2)
ax.plot_surface(X, Y, z_cbf)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.title.set_text("disc_dt{0}".format(delta_t))
ax.plot_surface(X, Y, z_disc)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.title.set_text("nn_dt{0}".format(delta_t))
ax.plot_surface(X, Y, z_nn)

ax.axes.set_xlim3d(left=-0.05, right=0.05) 
ax.axes.set_ylim3d(bottom=-0.1, top=0.1) 
ax.axes.set_zlim3d(bottom=-3, top=3) 

plt.savefig('input_plots/uloss_input_dt{0}.png'.format(str(delta_t)))
plt.clf()

# eval on control
print("Running system...")

system = inv_ped(10, 2, 1, -3, 3)
x = [0.01, -0.05]
t = np.linspace(0, 10, 100)
y_NN = system.run_system(x, t, NN_controller)
y_CBF = system.run_system(x, t, cbf_controller)
y_disc = system.run_system(x, t, disc_controller)
plt.xlim(-0.3, 0.3)
plt.ylim(-0.6, 0.6)
plt.plot(y_NN[:,0], y_NN[:, 1], label="NN")
plt.plot(y_CBF[:,0], y_CBF[:, 1], label="CBF")
plt.plot(y_disc[:,0], y_disc[:, 1], label="disc")
plt.xlabel('theta')
plt.ylabel('theta_dot')
plt.title('CBF vs NN vs disc')
plt.legend()
plt.savefig("controller_plots/uloss_controller_dt{0}.png".format(str(delta_t)))