# ================= 日志记录器 =================
import csv
import os
from datetime import datetime


class TrainingMonitor:
    def __init__(self, args, model_name):
        self.args = args
        os.makedirs("saved_logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"saved_logs/{model_name}_{timestamp}.csv"

        # 初始化CSV文件
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'Round', 'Loss', 'Accuracy',
                'Avg Client Time', 'Agg Time', 'Total Time']
            writer.writerow(header)

    def log(self, round_idx, loss, acc, client_times, agg_time, total_time):
        avg_client = sum(client_times) / len(client_times)

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_idx + 1,
                f"{loss:.4f}",
                f"{acc * 100:.2f}%",
                f"{avg_client:.2f}",
                f"{agg_time:.2f}",
                f"{total_time:.1f}"
            ])
