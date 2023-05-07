import random
import numpy as np
import torch
import torch.nn as nn
-


class Memory:
    def __init__(self, capacity):
        self.records = deque([], maxlen=capacity)

    def push(self, *args):
        record = Transition(*args)
        self.records.append(record)

    def sample(self, batch_size):
        return random.sample(self.records, batch_size)

    def __len__(self):
        return len(self.records)


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden1 = nn.Linear(state_dim, 30)
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2 = nn.Linear(30, 30)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, action_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = torch.relu(self.hidden1(state))
        x = torch.relu(self.hidden2(x))
        x = self.out(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim, gamma, memory_capacity, batch_size,
                 learning_rate, tau, epsilon_start, epsilon_end, epsilon_decay):
        self.name = "DQN"
        # 如果可以使用CUDA，就采用GPU运算，否则采用CPU运算
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s_dim, self.a_dim = state_dim, action_dim
        self.gamma = gamma  # 折扣系数
        self.memory_capacity = memory_capacity  # 经验容量
        self.memory = Memory(memory_capacity)
        self.batch_size = batch_size  # 经验回放采样量
        self.learning_rate = learning_rate  # 网络学习率
        self.tau = tau  # 软更新系数
        self.epsilon_start = epsilon_start  # 初始贪心系数
        self.epsilon_end = epsilon_end  # 最小贪心系数
        self.epsilon_decay = epsilon_decay # 贪心系数衰减值exp(-steps/epsilon_decay)

        self.policy = Net(state_dim, action_dim).to(self.device)
        self.target = Net(state_dim, action_dim).to(self.device)
        # 使初始目标网络和策略网络的参数相等
        self.target.load_state_dict(self.policy.state_dict())


        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate, amsgrad=True)

        self.steps = 0  # 经历的总步数，用于贪心系数的计算

    def choose_action(self, state):
        sample = random.random()
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1 * self.steps / self.epsilon_decay)
        self.steps += 1
        if sample > epsilon:
            with torch.no_grad():
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(0, self.a_dim)]], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def learn(self):
        # 如果当前经验数小于采样数，就不学习
        if len(self.memory) < self.batch_size:
            self.soft_update_network()
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 计算Huber损失函数
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化求解
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

        self.soft_update_network()

    def soft_update_network(self):
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = \
                policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_net_state_dict)
