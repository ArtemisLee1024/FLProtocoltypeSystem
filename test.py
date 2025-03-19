import requests

config = {
  "mode": "fedavg",
  "clients": 10,
  "rounds": 10,
  "local_epochs": 3,
  "dataset": "mnist",
  "model_config": {
    "layers": [
        {
            "type": "conv",
            "out_channels": 8,
            "kernel_size": 3,
            "padding": 1
        },
        {"type": "relu"},
        {"type": "pool", "kernel_size": 2},
        {
            "type": "conv",
            "out_channels": 16,
            "kernel_size": 3
        },
        {"type": "relu"},
        {"type": "pool", "kernel_size": 2},
        {"type": "flatten"},
        {
            "type": "fc",
            "out_features": 10
        }
    ]
  }
}

response = requests.get(
    'http://localhost:8080/api/history',
    # json=config,
    # stream=True
)

for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        # if decoded_line.startswith('METRICS|'):
        print(decoded_line)  # 输出训练日志
