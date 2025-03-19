import json
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, dataset, config_file):
        super().__init__()
        self.layers = nn.ModuleList()
        self._parse_config(dataset, config_file)

    def _parse_config(self, dataset, config_file):
        with open(config_file) as f:
            config = json.load(f)

        in_channels = 3 if dataset in ['cifar10'] else 1
        current_size = 32 if dataset in ['cifar10'] else 28
        in_features = None

        for layer_cfg in config['layers']:
            layer_type = layer_cfg['type']

            if layer_type == 'conv':
                self.layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'],
                    padding=layer_cfg.get('padding', 0)
                ))
                in_channels = layer_cfg['out_channels']
                current_size = current_size - layer_cfg['kernel_size'] + 1 + 2 * layer_cfg.get('padding', 0)

            elif layer_type == 'pool':
                self.layers.append(nn.MaxPool2d(
                    kernel_size=layer_cfg['kernel_size']
                ))
                current_size = current_size // layer_cfg['kernel_size']

            elif layer_type == 'fc':
                if in_features is None:
                    in_features = in_channels * current_size * current_size
                self.layers.append(nn.Linear(
                    in_features=in_features,
                    out_features=layer_cfg['out_features']
                ))
                in_features = layer_cfg['out_features']

            elif layer_type == 'relu':
                self.layers.append(nn.ReLU())

            elif layer_type == 'sigmoid':
                self.layers.append(nn.Sigmoid())

            elif layer_type == 'flatten':
                self.layers.append(nn.Flatten())

        self.classifier = self.layers[-1]
        self._print_model_info(config)

    def _print_model_info(self, config):
        print("\n========== 自定义模型配置 ==========")
        print("层配置：")
        for i, layer in enumerate(self.layers):
            print(f"层 {i + 1}: {str(layer)}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n总参数量：{total_params:,}")
        print("=" * 40)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_model(args):
    model = CustomModel(args.dataset, args.model_config)
    return model
