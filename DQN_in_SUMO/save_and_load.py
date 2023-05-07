import os
from datetime import datetime
import torch


def assign_train_directory(directory):
    now_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    directory_name = os.path.join(directory, now_time)
    os.mkdir(directory_name)
    return directory_name


def assign_test_directory(directory, model_saving_time):
    now_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    directory_name = os.path.join(directory, model_saving_time)
    if not os.path.isdir(directory_name):
        os.mkdir(directory_name)
    return directory_name, now_time

def save_model(agent, reward_data, directory):
    torch.save(agent.policy, os.path.join(directory, "policy_net.pth"))
    torch.save(agent.target, os.path.join(directory, "target_net.pth"))
    reward_data.to_csv(os.path.join(directory, "reward.csv"), index=False)


def load_model(model_saving_time):
    load_directory = os.path.join("output_model", model_saving_time, "policy_net.pth")
    policy_net = torch.load(load_directory)
    return policy_net