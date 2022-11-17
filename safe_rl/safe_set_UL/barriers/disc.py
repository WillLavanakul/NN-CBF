from infra import utils

class disc():
    def __init__(self, CBF, env, delta_t, num_inputs):
        self.CBF = CBF
        self.env = env
        self.delta_t = delta_t
        self.num_inputs = num_inputs
    
    def get_hyp(self, x):
        states = [x]
        inputs = utils.sample_inputs(states, self.num_inputs)
        input_labels = utils.label_inputs(states, inputs, self.env, self.CBF, self.delta_t)
        labels = utils.label_states(states, inputs, input_labels)
        a, b = labels[0]
        return a, b