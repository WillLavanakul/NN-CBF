import numpy as np
from sklearn.svm import LinearSVC
from controllers.constant import u_controller
import matplotlib.pyplot as plt

def sample_states(num_states, env, CBF):
  current_num = 0
  states = []
  while len(states) < num_states:
    x, y = np.random.uniform(-0.3, 0.3, 1)[0], np.random.uniform(-0.6, 0.6, 1)[0]
    if CBF.H(np.array([x, y])) >= 0:
      states.append([x, y])
  return np.array(states)

def sample_inputs(states, num_inputs):
  inputs = np.random.uniform(-3, 3, (len(states), num_inputs))
  inputs[:, 0] = -3.0
  inputs[:, 1] = 3.0
  return np.random.uniform(-3, 3, (len(states), num_inputs))

def label_inputs(states, inputs, env, CBF, delta_t):
  #print(len(states))
  labels = []
  for i in range(len(states)):
    label_u = []
    for u in inputs[i]:
      x_next = env.next_state(states[i], u, delta_t)
      label_u.append(np.sign(CBF.H(x_next)))
    labels.append(label_u)
  return np.array(labels)

def label_inputs_env(states, inputs, env, CBF, delta_t):
  labels = []
  for i in range(len(states)):
    label_u = []
    for u in inputs[i]:
      controller = u_controller(u)
      x_0 = states[i]
      y = env.run_system(x_0, delta_t, controller)
      x_final = y[-1, :]
      label_u.append(np.sign(CBF.H(x_final)))
    labels.append(label_u)
  return np.array(labels)


def label_states(states, inputs, input_labels):
  labels = []
  P = 0
  N = 0
  M = 0
  for i in range(len(input_labels)):
    if np.all(input_labels[i] == 1.0):
      P += 1
      a = 1
      b = -3
      labels.append([a, b])
    elif np.all(input_labels[i] == -1.0):
      N += 1
      a = 1
      b = 3
      labels.append([a, b])
    else:
      M += 1
      svc = fit_SVC(inputs[i], input_labels[i])
      a=svc.coef_[0]
      b=-svc.intercept_
      labels.append([a[0], b[0]])
  #print("P, M, N:", P, M, N)
  return np.array(labels)

def fit_SVC(inputs, labels):
  lsvc = LinearSVC(verbose=0)
  lsvc.fit(inputs.reshape(-1, 1), labels)
  return lsvc 

def get_data(num_data, num_inputs, delta_t, env, CBF):
  states = sample_states(num_data, env, CBF)
  inputs = sample_inputs(states, num_inputs)
  input_labels = label_inputs(states, inputs, env, CBF, delta_t)
  labels = label_states(states, inputs, input_labels)
  return states, labels, inputs, input_labels

def shuffle_data(data, labels, inputs, input_labels):
  idx = np.arange(len(data))
  np.random.shuffle(idx)
  return data[idx], labels[idx], inputs[idx], input_labels[idx]

def shuffle_data_u(data, inputs, input_labels):
  idx = np.arange(len(data))
  np.random.shuffle(idx)
  return data[idx], inputs[idx], input_labels[idx]