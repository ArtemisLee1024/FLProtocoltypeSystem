import copy

import torch
import torch.nn.functional as F


# ================= 联邦学习客户端核心逻辑 =================
class FLClient:
    def __init__(self, model, device, dp_params=None):
        self.model = model.to(device)
        self.device = device
        self.dp = dp_params

    def train(self, data_loader, local_epochs):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        for _ in range(local_epochs):
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()

                if self.dp:
                    self._add_dp_noise()

                optimizer.step()

        return copy.deepcopy(self.model.state_dict())
