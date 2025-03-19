import argparse
import copy
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torchvision import datasets, transforms

import custom_model
import fl_client
import fl_server
import training_monitor
import Models


# ================= 参数配置 =================
def parse_args():
    parser = argparse.ArgumentParser(description='联邦学习模拟器')
    parser.add_argument('--model_config', type=str, default=None,
                        help='自定义模型配置文件')
    parser.add_argument('--mode', choices=['fedavg', 'hierarchical'], default='fedavg',
                        help='训练模式: fedavg(传统联邦学习), hierarchical(分层聚合)')
    parser.add_argument('--hierarchical_groups', type=int, default=2,
                        help='分层聚合的分组数量（仅分层模式有效）')
    parser.add_argument('--clients', type=int, default=10, help='参与者数量')
    parser.add_argument('--rounds', type=int, default=20, help='全局训练轮次')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='客户端本地训练轮次')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'fashion'], default='mnist',
                        help='选择数据集: mnist/cifar10/fashion')
    parser.add_argument('--dp', action='store_true', help='启用差分隐私')
    parser.add_argument('--epsilon', type=float, default=1.0, help='DP参数ε')
    parser.add_argument('--delta', type=float, default=1e-5, help='DP参数δ')
    parser.add_argument('--encrypt', choices=['none', 'paillier', 'ckks'], default='none',
                        help='加密算法选择')
    parser.add_argument('--model_size', choices=['small', 'medium', 'large'], default=None,
                        help='模型规模选择')
    return parser.parse_args()


# ================= 模型选择器 =================
def create_model(args):
    if args.model_config:
        model = custom_model.create_model(args)
    elif args.model_size == 'small':
        model = Models.SmallModel(args.dataset)
    elif args.model_size == 'medium':
        model = Models.MediumModel(args.dataset)
    else:
        model = Models.LargeModel(args.dataset)
    print(f"\n当前模型结构：")
    print(model)
    return model


# ================= 数据准备 =================
def prepare_data(args):
    # 统一数据预处理
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:  # mnist/fashion
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # 加载数据集
    if args.dataset == 'mnist':
        train_data = datasets.MNIST('../../data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('../../data', train=False, transform=transform)
    elif args.dataset == 'cifar10':
        train_data = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('../../data', train=False, transform=transform)
    else:
        train_data = datasets.FashionMNIST('../../data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('../../data', train=False, transform=transform)

    # 分割客户端数据
    client_loaders = []
    indices = np.random.permutation(len(train_data))
    splits = np.array_split(indices, args.clients)
    for idx in splits:
        subset = torch.utils.data.Subset(train_data, idx)
        client_loaders.append(torch.utils.data.DataLoader(
            subset, batch_size=32, shuffle=True))

    # 测试数据集
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    return client_loaders, test_loader


# ================= 主程序 =================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    global_model = create_model(args).to(device)

    # 初始化日志记录器
    model_name = "custom" if args.model_config else args.model_size
    print(model_name)
    monitor = training_monitor.TrainingMonitor(args, f"{model_name}_{args.dataset}")

    # 准备数据
    client_loaders, test_loader = prepare_data(args)

    # 初始化服务器和客户端
    dp_params = {'epsilon': args.epsilon, 'delta': args.delta} if args.dp else None
    server = fl_server.FLServer(global_model, args)

    # 训练循环
    for round_idx in range(args.rounds):
        round_start = time.time()
        client_weights = []
        client_times = []

        # 客户端训练
        for loader in client_loaders:
            print("Client start train!", flush=True)
            client_start = time.time()
            client = fl_client.FLClient(copy.deepcopy(global_model), device, dp_params)
            weights = client.train(loader, args.local_epochs)
            client_times.append(time.time() - client_start)
            client_weights.append(weights)

        # 聚合更新
        agg_start = time.time()
        new_weights = server.aggregate(client_weights)
        agg_time = time.time() - agg_start
        global_model.load_state_dict(new_weights)

        # 评估
        test_loss, test_acc = server.evaluate(test_loader, device)
        # 输出统计信息
        total_time = time.time() - round_start
        avg_client_time = sum(client_times) / len(client_times)

        # 输出进度
        print(f"METRICS|"
              f"[Round {round_idx + 1}/{args.rounds}] "
              f"Loss: {test_loss:.4f} | Acc: {test_acc * 100:.2f}% | "
              f"Client Time: {avg_client_time:.2f}s/个 | "
              f"Agg Time: {agg_time:.2f}s | "
              f"Total: {total_time:.1f}s", flush=True)
        monitor.log(round_idx, test_loss, test_acc, client_times, agg_time, total_time)

    # ========== 新增模型保存逻辑 ==========
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_name}_{args.dataset}_{timestamp}.pth"
    save_path = os.path.join(save_dir, model_name)

    # 保存完整训练状态（包含模型参数、优化器状态等信息）
    torch.save({
        'model': global_model.state_dict(),
        'args': vars(args),  # 保存所有参数配置
        'timestamp': timestamp
    }, save_path)
    print(f"\n训练完成！模型已保存至：{save_path}", flush=True)


if __name__ == "__main__":
    main()
