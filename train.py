from GAIL import *
from agent import *
from utils import *

import rlgym
import os
import numpy as np

from configparser import ConfigParser
from argparse import ArgumentParser
import torch

#taken from AIGym will need to change to adapt to rlgym
env = rlgym.make(
    
)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

parser = ArgumentParser('parameters')
parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'ppo', help = 'actor training algorithm(default: ppo)')
parser.add_argument("--discriminator", type=str, default = 'gail', help = 'discriminator training algorithm(default: gail)')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: True)')

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

demonstrations_location_args = Dict(parser,'demonstrations_location',True)
agent_args = Dict(parser,args.agent)
discriminator_args = Dict(parser,args.discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None

discriminator = GAIL(writer, device, state_dim, action_dim, discriminator_args)
algorithm = PPO(device, state_dim, action_dim, agent_args)
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()

state_rms = RunningMeanStd(state_dim)
score_lst = []
discriminator_score_lst = []
score = 0.0
discriminator_score = 0

state_lst = []
state_ = (env.reset())
state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)

# begin training
for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            state_lst.append(state_)
            
            action, log_prob = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).to(device))
            
            next_state_, _, done, info = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            
            reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action).item()

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            discriminator_score += reward
            if done:
                state_ = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                discriminator_score = 0
            else:
                state = next_state
                state_ = next_state_
        agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi)
        state_rms.update(np.vstack(state_lst))
        state_lst = []
        if (n_epi % args.save_interval == 0 )& (n_epi != 0):
            torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))