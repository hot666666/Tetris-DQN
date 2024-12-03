import argparse
import os
from collections import deque
from random import random, randint, sample
import time
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from src.tetris import Tetris
from src.dqn import DQN


def get_args():
    parser = argparse.ArgumentParser("""Tetris 게임 환경 DQN 학습""")

    # 게임 환경 설정
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)

    # 하이퍼파라미터 설정
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)

    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--replay_memory_size", type=int, default=30000)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--exploration_fraction", type=float, default=0.25)

    parser.add_argument("--train_freq", type=int, default=20)

    parser.add_argument("--target_network", type=bool, default=False)
    parser.add_argument("--target_update_freq", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=1.0)

    # 저장 및 로깅 설정
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="models")

    args = parser.parse_args()
    return args


def linear_schedule(start: float, end: float, duration: int, t: int):
    slope = (end - start) / duration
    return max(slope * t + start, end)


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
    # scheduler =

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

    start_time = time.time()

    epoch = 0
    global_step = 0

    state = env.reset().to(device)
    while global_step < opt.total_timesteps:
        epsilon = linear_schedule(
            opt.initial_epsilon, opt.final_epsilon, opt.exploration_fraction * opt.total_timesteps, global_step)

        # 현재 상태에서 가능한 모든 행동들과 다음 상태들 가져오기
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        # Epsilon-greedy policy로 행동 선택
        if random() <= epsilon:  # Exploration
            index = randint(0, len(next_actions) - 1)
            next_state = next_states[index, :]
            action = next_actions[index]
        else:  # Exploitation
            with torch.no_grad():
                preds = model(next_states)[:, 0]
            index = torch.argmax(preds).item()
            next_state = next_states[index, :]
            action = next_actions[index]

        # 환경과 상호작용
        reward, done = env.step(action)
        global_step += 1

        # Replay memory에 저장
        replay_memory.append([state, reward, next_state, done])

        # 상태 업데이트
        if not done:
            state = next_state.to(device)
        else:
            state = env.reset().to(device)
            epoch += 1

        # Replay memory의 크기가 일정 이상이 되면 학습 시작
        if len(replay_memory) < max(opt.batch_size, minimum_replay_memory_size):
            continue

        # 학습 주기가 되어야 학습 시작
        if global_step % opt.train_freq != 0:
            continue

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

        # 기존 모델로 현재 상태에서 q-value 계산
        q_values = model(state_batch)

        # 타겟 모델로 다음 상태에서의 q-value를 계산하고 reward를 더해 target q-value를 계산
        with torch.no_grad():
            next_q_values = target_model(next_state_batch)
            done_batch = torch.tensor(done_batch, dtype=torch.float32)[
                :, None].to(device)
            target_q_values = reward_batch + opt.gamma * \
                next_q_values * (1 - done_batch)

        loss = F.mse_loss(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # Logging
        if global_step % 100 == 0:
            loss = loss.item()
            SPS = int(global_step / (time.time() - start_time))
            print(
                f"Epoch: {epoch}, Score: {env.score}, Cleared lines: {env.cleared_lines}")
            print(
                f"Global step: {global_step}, Loss: {loss:.4f}, SPS: {SPS}, Epsilon: {epsilon}")
            writer.add_scalar('train/TD_Loss', loss, global_step)
            writer.add_scalar(
                "train/q_value", q_values.mean().item(), global_step)
            writer.add_scalar("train/target_q_value",
                              target_q_values.mean().item(), global_step)
            writer.add_scalar("schedule/epsilon", epsilon, global_step)
            writer.add_scalar(
                "charts/SPS",
                SPS,
                global_step,
            )

        # Target Model 동기화
        if opt.target_network and global_step % opt.target_update_freq == 0:
            for target_model_param, model_param in zip(
                target_model.parameters(), model.parameters()
            ):
                target_model_param.data.copy_(
                    opt.tau * model_param.data
                    + (1.0 - opt.tau) * target_model_param.data
                )

        # Model save
        if epoch % opt.save_interval == 0 and opt.replay_memory_size == len(replay_memory):
            model_path = f"{opt.saved_path}/tetris_{epoch}"
            torch.save(model, model_path)
            print(f"Model saved at {model_path}")

    torch.save(model, f"{opt.saved_path}/tetris")


if __name__ == "__main__":
    opt = get_args()

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)

    train(opt)
