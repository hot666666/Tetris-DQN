import argparse
import os
from collections import deque
from random import random, randint, sample, choice
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
    parser.add_argument("--total_timesteps", type=int, default=100_000)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--replay_memory_size", type=int, default=50000)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=0.1)
    parser.add_argument("--exploration_fraction", type=float,
                        default=0.25)

    parser.add_argument("--train_freq", type=int, default=4)

    parser.add_argument("--target_network", type=bool, default=True)
    parser.add_argument("--target_update_freq", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=1.0)

    # 로깅 설정
    parser.add_argument("--exp_name", type=str,
                        default=os.path.basename(__file__)[: -len(".py")])
    parser.add_argument("--save_model_interval", type=int, default=200)
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project_name", type=str, default="Tetris-DQN")

    # 모델 저장 경로
    parser.add_argument("--saved_path", type=str, default="models")

    args = parser.parse_args()
    return args


def linear_schedule(start: float, end: float, duration: int, t: int):
    slope = (end - start) / duration
    return max(slope * t + start, end)

# 입실론 스케줄 함수


def epsilon_schedule(step, initial_epsilon, final_epsilon, exploration_steps):
    if step < exploration_steps:
        return initial_epsilon - (step / exploration_steps) * (initial_epsilon - final_epsilon)
    else:
        return final_epsilon


def train(opt):
    exploration_steps = int(opt.total_steps * 0.25)  # 입실론 감소 기간 (25% 스텝)
    initial_exploration_steps = 5000     # 초기 탐험 스텝 (완전 탐험)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Tensorboard
    writer = SummaryWriter(logdir="./runs")

    # Model, Optimizer, LR scheduler
    model = DQN().to(device)

    # DQN
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.00025, alpha=0.95, eps=1e-6)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

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

    start_time = time.time()

    epoch = 0
    global_step = 0

    state = env.reset().to(device)
    while global_step < opt.total_timesteps:
        # epsilon = linear_schedule(
        #     opt.initial_epsilon, opt.final_epsilon, opt.exploration_fraction * opt.total_timesteps, global_step)
        # 입실론 값 계산
        if global_step < initial_exploration_steps:
            epsilon = 1.0  # 초기 탐험 단계에서는 고정
        else:
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
        replay_memory.append([state, reward, next_state, done])

        # 상태 업데이트
        if not done:
            state = next_state.to(device)

        else:
            print(
                f"Epoch: {epoch}, Score: {env.score}, Cleared lines: {env.cleared_lines}")
            writer.add_scalar("epoch/score", env.score, epoch)
            state = env.reset().to(device)
            epoch += 1

        # Replay memory의 크기가 일정 이상이 되면 학습 시작
        if len(replay_memory) < opt.batch_size:
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

        # Logging
        if global_step % 100 == 0:
            loss = loss.item()
            SPS = int(global_step / (time.time() - start_time))
            print(
                f"Global step: {global_step}, Loss: {loss:.4f}, SPS: {SPS}, Epsilon: {epsilon:.2f}")
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
        if epoch % opt.save_model_interval == 0 and opt.replay_memory_size == len(replay_memory):
            model_path = f"{opt.saved_path}/tetris_{global_step}"
            torch.save(model, model_path)
            print(f"Model saved at {model_path}")

    torch.save(model, f"{opt.saved_path}/tetris")


if __name__ == "__main__":
    opt = get_args()

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    greek_letters = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    run_name = f"{opt.exp_name}/{choice(greek_letters)}_{choice(greek_letters)}__{int(time.time())}"

    # TensorBoard 로그 디렉토리 경로 설정
    os.environ["TENSORBOARD_LOGDIR"] = "./runs"

    if opt.wandb:
        import wandb

        run = wandb.init(
            project=opt.wandb_project_name,
            sync_tensorboard=True,
            config=vars(opt),
            name=run_name,
        )

    train(opt)
