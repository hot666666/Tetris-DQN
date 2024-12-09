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
    parser.add_argument("--num_epochs", type=int, default=5000)

    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--replay_memory_size", type=int, default=30000)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=int, default=1000)

    parser.add_argument("--target_network", type=bool, default=False)
    parser.add_argument("--target_update_freq", type=int, default=20)

    parser.add_argument("--train_freq", type=int, default=1)

    # 로깅 설정
    parser.add_argument("--exp_name", type=str,
                        default=os.path.basename(__file__)[: -len(".py")])
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project_name",
                        type=str, default="Tetris-DQN2")

    args = parser.parse_args()
    return args


# 입실론 스케줄 함수
def epsilon_schedule(step, initial_epsilon, final_epsilon, exploration_steps):
    if step < exploration_steps:  # 초기 탐험 스텝 (완전 탐험)
        return initial_epsilon - (step / exploration_steps) * (initial_epsilon - final_epsilon)
    else:
        return final_epsilon


def train(opt, log_dir, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    print(f"opt: {opt.__dict__}")

    minimum_replay_memory_size = opt.replay_memory_size // 10

    # Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # Model, Optimizer
    model = DQN().to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=opt.lr, alpha=0.95, eps=1e-6)

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

    max_cleared_lines = 10

    epoch = 0
    while epoch < opt.num_epochs:
        if epoch < opt.num_decay_epochs:
            epsilon = 1.0
        else:
            epsilon = epsilon_schedule(
                epoch - opt.num_decay_epochs, opt.initial_epsilon, opt.final_epsilon, opt.num_decay_epochs)

        state = env.reset().to(device)
        done = False
        while not done:
            # 현재 상태에서 가능한 모든 행동들과 다음 상태들을 가져옴
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

            # Replay memory에 저장
            replay_memory.append([state, reward, next_state, done])

            # 상태 업데이트
            if not done:
                state = next_state.to(device)
            else:
                print(
                    f"# Epoch: {epoch}, Score: {env.score}, Cleared lines: {env.cleared_lines}")
                if epoch > 0:
                    writer.add_scalar("epoch/score", env.score, epoch)
                    writer.add_scalar("epoch/cleared_lines",
                                      env.cleared_lines, epoch)

        # Replay memory가 충분히 쌓여야 학습 시작
        if len(replay_memory) < minimum_replay_memory_size:
            continue

        # 학습 주기가 되어야 학습 시작
        if epoch % opt.train_freq != 0:
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

        loss = F.mse_loss(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        loss = loss.item()
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Epsilon: {epsilon:.4f}")
        writer.add_scalar('train/td_loss', loss, epoch)
        writer.add_scalar("train/q_value", q_values.mean().item(), epoch)
        writer.add_scalar("train/target_q_value",
                          target_q_values.mean().item(), epoch)
        writer.add_scalar("schedule/epsilon", epsilon, epoch)

        # Model save
        if max_cleared_lines < env.cleared_lines:
            max_cleared_lines = env.cleared_lines
            model_path = f"models/{run_name}/tetris_{epoch}_{max_cleared_lines}"
            torch.save(model, model_path)
            print(f"Model saved at {model_path}")

    torch.save(model, f"models/{run_name}/tetris")


if __name__ == "__main__":
    opt = get_args()
    run_name = f"{opt.exp_name}/epo{opt.num_epochs}_decayEpo{opt.num_decay_epochs}_b{opt.batch_size}_rm{opt.replay_memory_size}_lr{opt.lr}__{int(time.time())}"

    # TensorBoard 로그 디렉토리 경로 설정
    log_dir = os.path.join("./runs", run_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    os.environ["TENSORBOARD_LOGDIR"] = log_dir

    # Model 저장 디렉토리 경로 설정
    if not os.path.isdir(f"./models/{run_name}"):
        os.makedirs(f"./models/{run_name}")

    if opt.wandb:
        import wandb

        run = wandb.init(
            project=opt.wandb_project_name,
            sync_tensorboard=True,
            config=vars(opt),
            name=run_name,
        )

    train(opt, log_dir, run_name)

    if opt.wandb:
        run.finish()
