from typing import List
import numpy as np

import torch
from torch import nn

from environments.environment_abstract import Environment, State
from utils import env_utils
from utils.misc_utils import evaluate_cost_to_go

import pickle

from to_implement.functions import get_nnet_model, train_nnet, value_iteration


def main():
    torch.set_num_threads(1)

    # get environment
    env: Environment = env_utils.get_environment("puzzle8")

    # get nnet model
    nnet: nn.Module = get_nnet_model()
    # get optimizer and lr scheduler
    optimizer = torch.optim.Adam(nnet.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.996)
    criterion = nn.MSELoss()

    device = torch.device('cpu')
    batch_size: int = 100
    num_itrs_per_vi_update: int = 200
    num_vi_updates: int = 50

    with open("sample_outputs/exercise_2_akash.txt", 'w') as f:
        # get data
        f.write("Preparing Data\n")
        data = pickle.load(open("data/data.pkl", "rb"))

        # train with supervised learning
        f.write("Training DNN\n")
        train_itr: int = 0
        for vi_update in range(num_vi_updates):
            f.write("--- Value Iteration Update: %i --- \n" % vi_update)
            states: List[State] = env.generate_states(
                batch_size*num_itrs_per_vi_update, (0, 500))

            states_nnet: np.ndarray = env.state_to_nnet_input(states)

            outputs_np = value_iteration(nnet, device, env, states)
            outputs = np.expand_dims(np.array(outputs_np), 1)

            nnet.train()
            train_nnet(nnet, states_nnet, outputs, batch_size,
                       num_itrs_per_vi_update, train_itr, criterion, optimizer, scheduler, f)

            nnet.eval()
            evaluate_cost_to_go(nnet, device, env,
                                data["states"], data["output"], f)

            train_itr = train_itr + num_itrs_per_vi_update


if __name__ == "__main__":
    main()
