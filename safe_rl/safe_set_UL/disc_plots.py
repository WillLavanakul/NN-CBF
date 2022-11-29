from barriers.FCN import FCN
from barriers.FCN_double import FCN_double
from barriers.FCN_tanh import FCN_tanh
from barriers.inv_ped_CBF import inv_CBF
from controllers.QP import QP_CBF_controller
from envs.inv_ped import inv_ped
from infra import utils
from barriers.disc import disc
import matplotlib.pyplot as plt
import numpy as np



c_a, a, b, m, g, l = 0.2, 0.075, 0.15, 2, 10, 1
u_min, u_max, x_min, x_max = -3, 3, [-0.3, -0.6], [0.3, 0.6]
cbf1 = inv_CBF(c_a, a, b, m, g, l)
env = inv_ped(g, m, l, u_min, u_max)

def train_model(delta_t):
    print("Starting training...")
    data, labels, all_inputs, all_input_labels = utils.get_data(1000, 200, delta_t, env, cbf1)
    train_size = 1000
    epochs = 50
    batch_size = 32
    train_data, train_labels = data[:train_size], labels[:train_size]
    model = FCN(2, 2, 4, 1000, 'relu', 5e-3)
    model.FCN.train()
    for epoch in range(epochs):
        if epoch % 10 == 0:
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
        if epoch % 10 == 0:
            print("Train loss:", running_loss_a, running_loss_b)
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

delta_t_s = np.linspace(0.0, 0.1, 20)
for d in range(len(delta_t_s)):
    delta_t = delta_t_s[d]
    print("delta_t:", delta_t)
    #model = train_model(delta_t)
    disc1 = disc(cbf1, env, delta_t, 5000)
    disc2 = disc(cbf1, env, delta_t, 5000, True)
    cbf_controller = QP_CBF_controller(target_input, cbf1)
    disc_controller1 = QP_CBF_controller(target_input, disc1)
    disc_controller2 = QP_CBF_controller(target_input, disc2)
    #nn_controller = QP_CBF_controller(target_input, model)

    # input plots
    n = 50
    x1 = np.linspace(-0.05, 0.05, n)
    x2 = np.linspace(-0.1, 0.1, n)
    z_cbf = -3.0*np.ones((n, n))
    z_disc = -3.0*np.ones((n, n))
    z_disc2 = -3.0*np.ones((n, n))
    for i in range(n):
        for j in range(n):
            x = np.array([x1[i], x2[j]])
            if cbf1.H(x) >= 0:
                a_cbf, b_cbf = cbf1.get_hyp(x)
                a_disc1, b_disc1 = disc1.get_hyp(x)
                a_disc2, b_disc2 = disc2.get_hyp(x)
                z_cbf[i, j] = b_cbf/a_cbf
                z_disc[i, j] = b_disc1/a_disc1
                z_disc2[i, j] = b_disc2/a_disc2
                #z_nn[i, j] = u_nn
    
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
    ax.plot_surface(X, Y, z_disc2)



    plt.savefig('disc_plots/input/disc_dt{0}.png'.format(delta_t))
    plt.clf()

    # print("Running system...")

    # system = inv_ped(10, 2, 1, -3, 3)
    # x = [0.01, -0.05]
    # t = np.linspace(0, 10, 100)
    # y_CBF = system.run_system(x, t, cbf_controller)
    # y_disc1 = system.run_system(x, t, disc_controller1)
    # y_disc2 = system.run_system(x, t, disc_controller2)

    # plt.xlim(-0.3, 0.3)
    # plt.ylim(-0.6, 0.6)
    # plt.plot(y_CBF[:, 0], y_CBF[:, 1], label="CBF")
    # plt.plot(y_disc1[:, 0], y_disc1[:, 1], label='disc1')
    # plt.plot(y_disc2[:, 0], y_disc2[:, 1], label='disc2')
    # plt.xlabel('theta')
    # plt.ylabel('theta_dot')
    # plt.title('CBF vs euler vs sim')
    # plt.legend()
    # plt.savefig('disc_plots/control/disc_dt{0}.png'.format(str(delta_t)))