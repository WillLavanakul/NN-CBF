import numpy as np
from sklearn.svm import LinearSVC
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

def label_states(states, inputs, input_labels):
  labels = []
  for i in range(len(input_labels)):
    if np.all(input_labels[i] == 1.0):
      a = 1
      b = -3
      labels.append([a, b])
    elif np.all(input_labels[i] == -1.0):
      a = 1
      b = 3
      labels.append([a, b])
    else:
      svc = fit_SVC(inputs[i], input_labels[i])
      a=svc.coef_[0]
      b=-svc.intercept_
      labels.append([a[0], b[0]])
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

def shuffle_data(data, labels):
  idx = np.arange(len(data))
  np.random.shuffle(idx)
  return data[idx], labels[idx]

def shuffle_data_u(data, inputs, input_labels):
  idx = np.arange(len(data))
  np.random.shuffle(idx)
  return data[idx], inputs[idx], input_labels