import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils.utils import ReplayBuffer, make_one_mini_batch, convert_to_tensor

class Agent(nn.Module):
    def __init__(self,algorithm, writer, device, state_dim, action_dim, args, demonstrations_location_args): 
        super(Agent, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.data = ReplayBuffer(action_prob_exist = True, max_size = self.args.traj_length, state_dim = state_dim, num_action = action_dim)
        file_size = 120
        
        f = open(demonstrations_location_args.expert_state_location,'rb')
        self.expert_states = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float()

        # f = open(demonstrations_location_args.expert_action_location,'rb')
        # self.expert_actions = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)]))
        
        f.close()
        
        self.brain = algorithm
        
    def get_action(self,x):
        action, log_prob = self.brain.get_action(x)
        return action, log_prob
    
    def put_data(self,transition):
        self.data.put_data(transition)
    
    def train(self, discriminator, discriminator_batch_size, state_rms, n_epi, batch_size = 64):
        if self.args.on_policy :
            data = self.data.sample(shuffle = False)
            states, actions, rewards, next_states, done_masks, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])
        
        # agent_s,agent_a = make_one_mini_batch(discriminator_batch_size, states, actions)
        agent_s, agent_a = make_one_mini_batch(discriminator_batch_size, states)
        # expert_s,expert_a = make_one_mini_batch(discriminator_batch_size, self.expert_states, self.expert_actions)
        expert_s, expert_a = make_one_mini_batch(discriminator_batch_size, self.expert_states)
        if self.args.on_policy :
            expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        
        # discriminator.train_network(self.writer, n_epi, agent_s, agent_a, expert_s, expert_a)
        discriminator.train_network(self.writer, n_epi, agent_s, None, expert_s, None)

        if self.args.on_policy :
            self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks, old_log_probs)