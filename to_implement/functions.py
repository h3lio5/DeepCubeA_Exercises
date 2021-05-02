from typing import List
import torch
import torch.nn as nn
import numpy as np
from environments.environment_abstract import Environment, State
from utils.misc_utils import *


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    net = nn.Sequential(nn.Linear(81, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(), nn.Linear(
        128, 256), nn.BatchNorm1d(num_features=256), nn.ReLU(), nn.Linear(256, 1))
    return net


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nnet.parameters(), lr=3e-5)

    for step in range(num_itrs):
        # Iterate over batches of data
        states = torch.FloatTensor(
            states_nnet[step*batch_size:(step+1)*batch_size])
        target = torch.FloatTensor(
            outputs[step*batch_size:(step+1)*batch_size])
        net_output = nnet(states)
        loss = criterion(target, net_output)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(nnet.parameters(), 3)
        optimizer.step()

        if step % 100 == 0:
            print(
                f'Itr: {train_itr + step}, loss: {loss.item()}, targ_ctg: {np.mean(target.data.numpy())}, nnet_ctg: {np.mean(net_output.data.numpy())}')


def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    # total number of states
    num_states = len(states)
    # get the next states and the corresponding transition costs
    next_states, trans_costs = env.expand(states)
    flattened_nx_states, _ = flatten(next_states)
    flattened_tc, _ = flatten(trans_costs)
    # convert states into numpy arrays
    flattened_nx_states_input = env.state_to_nnet_input(flattened_nx_states)
    outputs = []
    with torch.no_grad():
        # Taking the batch_size to be 1000
        for step in range(len(flattened_nx_states_input)//1000):
            inputs = torch.FloatTensor(
                flattened_nx_states_input[step*1000:(step+1)*1000])
            output = nnet(inputs)
            outputs.extend(output.numpy())
    # updated cost-to-go
    # import pdb
    # pdb.set_trace()
    outputs_np = np.array(outputs).squeeze()
    flattened_tc_np = np.array(flattened_tc)
    J_new = outputs_np + flattened_tc_np
    # Group the J values based on their respcetive input/origin states
    J_new = np.array_split(J_new, num_states)
    # perform Bellman Update
    J_new = list(map(lambda x: min(x), J_new))

    return J_new
