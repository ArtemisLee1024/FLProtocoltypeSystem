import numpy as np
import tenseal as ts
import torch
import torch.nn.functional as F
from phe import paillier


# ================= 联邦学习聚合服务器核心逻辑 =================
class FLServer:
    def __init__(self, global_model, args):
        self.global_model = global_model
        self.args = args
        self._init_encryption()

    def _init_encryption(self):
        if self.args.encrypt == 'paillier':
            self.pub_key, self.priv_key = paillier.generate_paillier_keypair()
        elif self.args.encrypt == 'ckks':
            self.ckks_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.ckks_context.generate_galois_keys()
            self.ckks_context.global_scale = 2 ** 40

    def aggregate(self, client_weights):
        if self.args.encrypt != 'none':
            return self._encrypted_aggregation(client_weights)

        if self.args.mode == 'fedavg':
            return self._fedavg(client_weights)
        elif self.args.mode == 'hierarchical':
            return self._hierarchical_agg(client_weights)

    def _fedavg(self, client_weights):
        averaged_weights = {}
        for key in client_weights[0].keys():
            averaged_weights[key] = torch.mean(
                torch.stack([w[key] for w in client_weights]), dim=0)
        return averaged_weights

    def _hierarchical_agg(self, client_weights):
        group_size = len(client_weights) // self.args.hierarchical_groups
        groups = [client_weights[i * group_size:(i + 1) * group_size]
                  for i in range(self.args.hierarchical_groups)]

        # 第一层组内聚合
        group_models = [self._fedavg(g) for g in groups]

        # 第二层全局聚合
        return self._fedavg(group_models)

    def _encrypted_aggregation(self, client_weights):
        if self.args.encrypt == 'paillier':
            return self._paillier_aggregation(client_weights)
        elif self.args.encrypt == 'ckks':
            return self._ckks_aggregation(client_weights)

    def _paillier_aggregation(self, client_weights):
        aggregated_weights = {}
        for key in client_weights[0].keys():
            shape = client_weights[0][key].shape
            flatten_weights = [w[key].flatten().numpy() for w in client_weights]

            encrypted_sum = [self.pub_key.encrypt(0.0)] * len(flatten_weights[0])
            for vec in flatten_weights:
                encrypted_vec = [self.pub_key.encrypt(float(x)) for x in vec]
                encrypted_sum = [x + y for x, y in zip(encrypted_sum, encrypted_vec)]

            decrypted = [self.priv_key.decrypt(x) for x in encrypted_sum]
            averaged = torch.tensor(decrypted).view(shape) / len(client_weights)
            aggregated_weights[key] = averaged
        return aggregated_weights

    def _ckks_aggregation(self, client_weights):
        aggregated_weights = {}
        for key in client_weights[0].keys():
            max_len = max(w[key].numel() for w in client_weights)
            padded_weights = []
            for w in client_weights:
                vec = w[key].flatten().numpy()
                padded = np.pad(vec, (0, max_len - len(vec)))
                padded_weights.append(padded)

            encrypted_sum = ts.ckks_vector(self.ckks_context, np.zeros(max_len))
            for vec in padded_weights:
                encrypted_vec = ts.ckks_vector(self.ckks_context, vec.tolist())
                encrypted_sum += encrypted_vec

            decrypted = encrypted_sum.decrypt()
            averaged = np.array(decrypted) / len(client_weights)
            original_shape = client_weights[0][key].shape
            aggregated_weights[key] = torch.tensor(
                averaged[:original_shape.numel()]).view(original_shape).float()
        return aggregated_weights

    def evaluate(self, test_loader, device):
        self.global_model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = self.global_model(x)
                total_loss += F.cross_entropy(output, y, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(y).sum().item()

        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        return avg_loss, accuracy
