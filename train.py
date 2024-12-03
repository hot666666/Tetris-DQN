import argparse
import os
from collections import deque
from random import random, randint, sample

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from src.tetris import Tetris
from dqn import DQN


def get_args():
    parser = argparse.ArgumentParser("""Tetris2024 게임 환경 DQN 학습""")

    # 게임 환경 설정
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)

    # 하이퍼파라미터 설정
    parser.add_argument("--num_epochs", type=int, default=5000)

    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--replay_memory_size", type=int, default=30000)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=int, default=2000)

    parser.add_argument("--target_network", type=bool, default=False)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    # parser.add_argument("--tau", type=float, default=0.005)

    # 저장 및 로깅 설정
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="models")

    args = parser.parse_args()
    return args


def epsilon_greedy_policy(predictions, num_actions, epsilon) -> int:
    if random() <= epsilon:  # Exploration
        return randint(0, num_actions - 1)
    else:  # Exploitation
        return torch.argmax(predictions).item()


def set_dir(opt):
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Tensorboard
    writer = SummaryWriter(opt.log_path)

    # Model, Optimizer, LR scheduler, Loss function
    model = DQN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=opt.replay_memory_size//opt.batch_size+1, eta_min=1e-5)

    # Target model
    if opt.target_network:
        target_model = DQN().to(device)
        target_model.load_state_dict(model.state_dict())
    else:
        target_model = model

    # Environment
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    # Replay memory
    replay_memory = deque(maxlen=opt.replay_memory_size)
    minimum_replay_memory_size = opt.replay_memory_size // 10

    max_cleared_lines = 0

    # Epoch = 학습이 진행된 Episode
    epoch = 0
    while epoch < opt.num_epochs:
        # epsilon with linear decay
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
            opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

        state = env.reset().to(device)
        done = False
        while not done:
            # 현재 상태에서 가능한 모든 행동들과 다음 상태들을 가져옴
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())

            # 다음 상태들에 대한 q-value를 계산하고, 이를 바탕으로 현재 상태에서 최적의 행동을 선택(또는 랜덤하게 행동을 선택)
            next_states = torch.stack(next_states).to(device)
            with torch.no_grad():
                predictions = model(next_states)[:, 0]

            # 행동 선택은 epsilon-greedy policy을 따름
            index = epsilon_greedy_policy(
                predictions, len(next_steps), epsilon)
            next_state = next_states[index, :]
            action = next_actions[index]

            # 환경과 상호작용
            reward, done = env.step(action, render=False)

            # Replay memory에 저장
            replay_memory.append([state, reward, next_state, done])

            # 상태 업데이트
            if not done:
                state = next_state.to(device)

        # Replay memory가 충분히 쌓여야 학습 시작
        if len(replay_memory) < max(opt.batch_size, minimum_replay_memory_size):
            continue

        # 학습 시작
        epoch += 1

        # Batch sampling
        batch = sample(replay_memory, opt.batch_size)
        state_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch)
        state_batch = torch.stack(
            tuple(state for state in state_batch)).to(device)
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.int32)[:, None]).to(device)
        next_state_batch = torch.stack(
            tuple(state for state in next_state_batch)).to(device)

        # 현재 상태에서의 행동에 대한 q-value
        q_values = model(state_batch)

        # 다음 상태에서의 q-value를 계산하고, target q-value를 계산
        with torch.no_grad():
            next_q_values = target_model(next_state_batch)
            done_batch = torch.tensor(done_batch, dtype=torch.float32)[
                :, None].to(device)
            target_q_values = reward_batch + opt.gamma * \
                next_q_values * (1 - done_batch)

        optimizer.zero_grad()
        loss = F.mse_loss(q_values, target_q_values)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Target network 동기화
        if opt.target_network and epoch % opt.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Logging
        formatted_loss = float(f"{loss.item():.4f}")
        print(
            f"Epoch: {epoch}/{opt.num_epochs}, Loss: {formatted_loss}, Score: {env.score}, Cleared lines: {env.cleared_lines}")
        writer.add_scalar('Train/Loss', formatted_loss, epoch)
        writer.add_scalar('Train/Score', env.score, epoch)
        writer.add_scalar('Train/Cleared lines', env.cleared_lines, epoch)
        writer.add_scalar("Train/q_value", q_values.mean().item(), epoch)

        # Model save
        if epoch % opt.save_interval == 0 and opt.replay_memory_size == len(replay_memory):
            model_path = f"{opt.saved_path}/tetris_{epoch}"
            torch.save(model, model_path)
            print(f"Model saved at {model_path}")

        if env.cleared_lines > 10000 and env.cleared_lines > max_cleared_lines:
            max_cleared_lines = env.cleared_lines
            torch.save(
                model, f"{opt.saved_path}/tetris_best_{max_cleared_lines}")

    torch.save(model, f"{opt.saved_path}/tetris")


if __name__ == "__main__":
    opt = get_args()
    set_dir(opt)

    train(opt)
