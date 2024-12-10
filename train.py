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
    parser.add_argument("--total_timesteps", type=int, default=500_000)

    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--replay_memory_size", type=int, default=30000)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--exploration_fraction", type=float,
                        default=0.4)

    parser.add_argument("--train_freq", type=int, default=4)

    # 타겟 네트워크 설정(time-step 단위)
    parser.add_argument("--target_network", type=bool, default=True)
    parser.add_argument("--target_update_freq", type=int, default=1000)

    # 로깅 설정
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project_name",
                        type=str, default="Tetris-DQN")
    parser.add_argument("--exp_name", type=str,
                        default=os.path.basename(__file__)[: -len(".py")])
    parser.add_argument("--log_interval", type=int, default=1000)

    # 모델 저장
    parser.add_argument("--save_model_interval", type=int, default=2000)

    args = parser.parse_args()
    return args


# 입실론 스케줄 함수
def epsilon_schedule(step, initial_epsilon, final_epsilon, exploration_steps):
    # 선형적으로 감소
    if step < exploration_steps:
        epsilon = initial_epsilon - \
            (initial_epsilon - final_epsilon) * (step / exploration_steps)
    else:
        epsilon = final_epsilon
    return epsilon


def train(opt, log_dir, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    print(f"opt: {opt.__dict__}")

    # Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Tensorboard
    writer = SummaryWriter(logdir=log_dir)

    # Model, Optimizer, LR scheduler
    model = DQN().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)
    loss_fn = F.mse_loss

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

    exploration_steps = int(opt.total_timesteps * opt.exploration_fraction)

    epoch = 0
    global_step = 0
    max_cleared_lines = 10

    state = env.reset().to(device)
    while global_step < opt.total_timesteps:
        epsilon = epsilon_schedule(
            global_step, opt.initial_epsilon, opt.final_epsilon, exploration_steps)

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
        next_state = next_state.to(device)
        replay_memory.append([state, reward, next_state, done])

        # 상태 업데이트
        if not done:
            state = next_state
        else:
            print(
                f"Epoch: {epoch}, Score: {env.score}, Cleared lines: {env.cleared_lines}")
            writer.add_scalar("epoch/score", env.score, global_step)
            writer.add_scalar("epoch/cleared_lines",
                              env.cleared_lines, global_step)
            state = env.reset().to(device)
            epoch += 1

        # Replay memory의 크기가 일정 이상이 되면 학습 시작
        if len(replay_memory) < opt.batch_size:
            continue

        # 학습 주기가 되어야 학습 시작
        if global_step % opt.train_freq != 0:
            continue

        # 배치 샘플링
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

        loss = loss_fn(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if global_step % opt.log_interval == 0:
            loss = loss.item()
            print(
                f"Global step: {global_step}, Loss: {loss:.4f}, Epsilon: {epsilon:.2f}")
            writer.add_scalar('train/td_loss', loss, global_step)
            writer.add_scalar(
                "train/q_value", q_values.mean().item(), global_step)
            writer.add_scalar("train/target_q_value",
                              target_q_values.mean().item(), global_step)
            writer.add_scalar("schedule/epsilon", epsilon, global_step)

        # Target Model 동기화
        if opt.target_network and global_step % opt.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Best model save
        if env.cleared_lines > max_cleared_lines:
            max_cleared_lines = env.cleared_lines
            model_path = f"models/{run_name}/tetris_best_{max_cleared_lines}"
            torch.save(model, model_path)
            print(f"Best Model saved at {model_path}")
            continue

        # Model save
        if global_step % opt.save_model_interval == 0:
            model_path = f"models/{run_name}/tetris_{global_step}"
            torch.save(model, model_path)
            print(f"Model saved at {model_path}")

    torch.save(model, f"{opt.saved_path}/tetris")


if __name__ == "__main__":
    opt = get_args()

    run_name = f"{opt.exp_name}/{opt.batch_size}_{opt.replay_memory_size}_{opt.lr}_{opt.target_update_freq}__{int(time.time())}"

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
