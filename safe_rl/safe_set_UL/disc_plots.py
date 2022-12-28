from barriers.FCN import FCN
from barriers.FCN_double import FCN_double
from barriers.FCN_log import FCN_norm
from barriers.FCN_u import FCN_u
from barriers.inv_ped_CBF import inv_CBF
from controllers.QP import QP_CBF_controller
from envs.inv_ped import inv_ped
from infra import utils
from barriers.disc import disc
import matplotlib.pyplot as plt
import numpy as np


delta_t_sys = 0.05
c_a, a, b, m, g, l = 0.2, 0.075, 0.15, 2, 10, 1
u_min, u_max, x_min, x_max = -3, 3, [-0.3, -0.6], [0.3, 0.6]
cbf1 = inv_CBF(c_a, a, b, m, g, l)
env = inv_ped(g, m, l, u_min, u_max, delta_t_sys)

def train_model(delta_t):
    print("Starting training...")
    train_size = 10000
    input_size = 100
    train_data, train_labels, inputs, input_labels = utils.get_data(train_size, input_size, delta_t, env, cbf1)
    epochs = 500
    batch_size = 320
    finetine = -1
    model = FCN_double(2, 2, 5, 1000, 'relu', 5e-4)
    model.FCN_a.train()
    model.FCN_b.train()
    for epoch in range(epochs):
        if epoch % 10 == 0:
            print("Epoch:", epoch)
        s_train_data, s_train_labels, s_inputs, s_input_labels = utils.shuffle_data(train_data, train_labels, inputs, input_labels)
        running_loss_a = 0
        running_loss_b = 0
        for batch in range(train_size // batch_size):
            batch_data = s_train_data[batch*batch_size:batch*batch_size+batch_size]
            if epoch < finetine:
              batch_labels = s_train_labels[batch*batch_size:batch*batch_size+batch_size]
              batch_loss_a, batch_loss_b = model.update(batch_data, batch_labels)
            else:
              batch_inputs = s_inputs[batch*batch_size:batch*batch_size+batch_size]
              batch_input_labels = s_input_labels[batch*batch_size:batch*batch_size+batch_size]
              batch_loss_a, batch_loss_b = model.update2(batch_data, batch_inputs, batch_input_labels)
            running_loss_a += batch_loss_a
            running_loss_b += batch_loss_b
        if epoch % 10 == 0:
            print("Train loss:", running_loss_a, running_loss_b)

            figure, axis = plt.subplots(5, 1)
            for i in range(5):
              x_v = train_data[i]
              a_v, b_v = model.get_hyp(x_v)
              axis[i].scatter(inputs[i], input_labels[i], s=1, label='inputx')
              a_t, b_t = train_labels[i][0], train_labels[i][1]
              axis[i].scatter(b_t/a_t, -0.25, s=2, label='SVM')
              u_plot = np.arange(-3, 3, 0.1)
              axis[i].fill_between(u_plot, -0.25,
                 where = (a_t*u_plot-b_t > 0),
                 color = 'y',
                 alpha=0.2)

              axis[i].scatter(b_v/a_v, 0, s=1, label='NN')
              axis[i].fill_between(u_plot, 0,
                 where = (a_v*u_plot-b_v > 0),
                 color = 'g',
                 alpha=0.2)
            plt.legend()
            plt.savefig("disc_plots/hyperplanes/hyperplane{0}.png".format(epoch))

            plt.clf()
    return model

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

#delta_t_s = np.linspace(0.04, 0.06, 3)
delta_t_s = [0.05]
for d in range(len(delta_t_s)):
    delta_t = delta_t_s[d]
    print("delta_t:", delta_t)
    model = train_model(delta_t)
    disc1 = disc(cbf1, env, delta_t, 500)
    disc2 = disc(cbf1, env, delta_t, 500, True)
    cbf_controller = QP_CBF_controller(target_input, cbf1)
    disc_controller1 = QP_CBF_controller(target_input, disc1)
    disc_controller2 = QP_CBF_controller(target_input, disc2)
    nn_controller = QP_CBF_controller(target_input, model)

    # input plots
    n = 50
    x1 = np.linspace(-0.05, 0.05, n)
    x2 = np.linspace(-0.1, 0.1, n)
    z_cbf = -3.0*np.ones((n, n))
    z_disc = -3.0*np.ones((n, n))
    z_nn = -3.0*np.ones((n, n))
    for i in range(n):
        for j in range(n):
            x = np.array([x1[i], x2[j]])
            if cbf1.H(x) >= 0:
                # a_cbf, b_cbf = cbf1.get_hyp(x)
                # a_disc1, b_disc1 = disc1.get_hyp(x)
                # a_nn, b_nn = model.get_hyp(x)
                # z_cbf[i, j] = b_cbf/a_cbf
                # z_disc[i, j] = b_disc1/a_disc1
                # z_nn[i, j] = b_nn / a_nn
                z_cbf[i, j] = cbf_controller.forward(x, 0)
                z_disc[i, j] = disc_controller1.forward(x, 0)
                z_nn[i, j] = nn_controller.forward(x, 0)
    
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.title.set_text("cbf")
    
    ax.axes.set_xlim3d(left=-0.05, right=0.05) 
    ax.axes.set_ylim3d(bottom=-0.1, top=0.1) 
    ax.axes.set_zlim3d(bottom=-3, top=3) 

    X, Y = np.meshgrid(x1, x2)
    ax.plot_surface(X, Y, z_cbf)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.title.set_text("disc_dt{0}".format(delta_t))
    
    ax.axes.set_xlim3d(left=-0.05, right=0.05) 
    ax.axes.set_ylim3d(bottom=-0.1, top=0.1) 
    ax.axes.set_zlim3d(bottom=-3, top=3) 

    ax.plot_surface(X, Y, z_disc)



    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.title.set_text("sim{0}".format(delta_t))

    ax.axes.set_xlim3d(left=-0.05, right=0.05) 
    ax.axes.set_ylim3d(bottom=-0.1, top=0.1) 
    ax.axes.set_zlim3d(bottom=-3, top=3) 
    ax.plot_surface(X, Y, z_nn)



    plt.savefig('disc_plots/input/disc_dt{0}.png'.format(delta_t))
    plt.clf()

    print("Running system...")

    system = inv_ped(10, 2, 1, -3, 3, delta_t)
    x = np.array([0.01, -0.05])
    t = 10
    y_CBF = system.run_system(x, t, cbf_controller)
    y_disc1 = system.run_system(x, t, disc_controller1)
    #y_disc2 = system.run_system(x, t, disc_controller2)
    y_NN = system.run_system(x, t, nn_controller)

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.6, 0.6)
    plt.plot(y_CBF[:, 0], y_CBF[:, 1], label="CBF")
    plt.plot(y_NN[:, 0], y_NN[:, 1], label='NN')
    #plt.plot(y_disc2[:, 0], y_disc2[:, 1], label='disc2')
    plt.plot(y_disc1[:, 0], y_disc1[:, 1], label='disc1')
    plt.xlabel('theta')
    plt.ylabel('theta_dot')
    plt.title('CBF vs euler vs sim')
    plt.legend()
    plt.savefig('disc_plots/control/disc_dt{0}.png'.format(str(delta_t)))
    plt.clf()