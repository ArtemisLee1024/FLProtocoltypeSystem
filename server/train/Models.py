import torch.nn as nn


# ================= 神经网络模型 =================
class SmallModel(nn.Module):
    """参数量约50K"""

    def __init__(self, dataset):
        super().__init__()
        # 根据数据集调整输入参数
        in_channels = 3 if dataset == 'cifar10' else 1
        input_size = 3072 if dataset == 'cifar10' else 784

        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MediumModel(nn.Module):
    """参数量约1.2M"""

    def __init__(self, dataset):
        super().__init__()
        in_channels = 3 if dataset == 'cifar10' else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 7 * 7 if dataset == 'cifar10' else 64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LargeModel(nn.Module):
    """参数量约5.8M"""
    def __init__(self, dataset):
        super().__init__()
        in_channels = 3 if dataset == 'cifar10' else 1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
