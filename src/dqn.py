import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        # 게임 상태에 대한 휴리스틱한 정보(지워진 줄, 구멍, 인접열 차이 합, 높이 합)를 입력으로 받아서, q-value를 출력하는 DQN 모델
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
