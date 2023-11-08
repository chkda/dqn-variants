import os
import time
import random
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import NoopResetEnv, FireResetEnv, ClipRewardEnv, EpisodicLifeEnv, MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def make_env(env_id, seed, idx, capture_video, run_name):
    def env_creator():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, 4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = ClipRewardEnv(env)
        env.action_space.seed(seed)
        return env
    return env_creator

class DuelingDQN(nn.Module):

    def __init__(self, action_dims: int):
        super().__init__()
        self.action_dims = action_dims
        self.shared_network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.advantage_network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dims),
        )
        self.value_network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, inp):
        x = self.shared_network(inp/255)
        adv = self.advantage_network(x)
        value = self.value_network(x)
        value = value.expand(x.size(0), self.action_dims)
        q_value = value + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_dims)
        return q_value


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(t * slope + start_e, end_e)


def train(config):
    seed = config["seed"]
    run_name = f"{config['env']}__{config['exp_name']}__{seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(config["env"], seed + i, i, True, run_name) for i in range(config["num_envs"])]
    )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )
    action_dims = envs.single_action_space.n
    q_network = DuelingDQN(action_dims).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    target_network = DuelingDQN(action_dims).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        config["buffer_size"],
        envs.single_observation_space,
        envs.single_action_space,
        device,
        config["num_envs"],
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    obs, _ = envs.reset(seed=seed)
    for global_step in range(config["timesteps"]):
        epsilon = linear_schedule(config["start_e"], config["end_e"], config["exploration_fraction"] * config["timesteps"], global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if 'final_info' in infos:
            for info in infos['final_info']:
                if 'episode' not in info:
                    continue
                writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
                writer.add_scalar("charts/episodic_length", info['episode']['l'], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos['final_observation'][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > config['learning_starts']:
            if global_step % config['train_frequency'] == 0:
                data = rb.sample(config['batch_size'])
                with torch.no_grad():
                    _, target_actions = target_network(data.next_observations).max(dim=1)
                    target_actions = target_actions.unsqueeze(dim=1)
                    target_max = q_network(data.next_observations).gather(1, target_actions).squeeze()
                    td_target = data.rewards.flatten() + config['gamma'] * target_max * (1 - data.dones.flatten())

                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(old_val, td_target)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_losses", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % config['target_network_frequency'] == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        tau = config['tau']
                        target_network_param.data.copy_(
                            tau * q_network_param.data + (1 -tau) * target_network_param.data
                        )

    envs.close()
    writer.close()

    return


config = {
    'seed': 1,
    'env': 'ALE/Breakout-v5',
    'timesteps': 10000000,
    'num_envs': 1,
    'buffer_size': 1000000,
    'gamma': 0.99,
    'tau': 1,
    'target_network_frequency': 1000,
    'batch_size': 32,
    'start_e': 1,
    'end_e':  0.01,
    'exploration_fraction': 0.1,
    'learning_starts': 80000,
    'train_frequency': 4,
    'exp_name': 'Breakout',
    'learning_rate': 0.0001,
}


train(config)
