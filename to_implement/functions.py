from typing import List
import torch
import torch.nn as nn
import numpy as np
from environments.environment_abstract import Environment, State


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    net = nn.Sequential(nn.Linear(81, 250), nn.ReLU(), nn.Linear(
        250, 250), nn.ReLU(), nn.Linear(250, 1))
    return net


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nnet.parameters(), lr=1e-3)
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
        optimizer.step()
        if step % 100 == 0:
            print(
                f'Itr: {step}, loss: {loss.item()}, targ_ctg: {np.mean(target.data.numpy())}, nnet_ctg: {np.mean(net_output.data.numpy())}')


def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    pass
