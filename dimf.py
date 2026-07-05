import os
import re
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import KR_norm_juicer as KRnorm


# ==========================================
# 0. 早停止辅助类 (Early Stopping)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None

    def step(self, loss, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"[Early Stop Monitor] Accumulated counts: {self.counter} / {self.patience} (Optimal Loss: {self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ==========================================
# 1. 基础参数与工具类
# ==========================================
class Parameters:
    def __init__(self, n=None, K=3, p=5, b=2.0, overlap_threshold=0.7, resolution=40000):
        self.n = n
        self.K = K
        self.p = p
        self.b = b
        self.overlap_threshold = overlap_threshold
        self.resolution = resolution

        # 物理先验参数
        self.min_tad_bins = 6 if resolution <= 25000 else 5
        self.max_tad_bins = 60
        self.min_parent_to_split = 50
        self.max_gap_to_close = 0


class ClusterUtils:
    @staticmethod
    def find_all_consecutive_ones(binary_array, min_len=5, max_len=None):
        intervals = []
        current_start = -1
        current_length = 0
        for i in range(len(binary_array)):
            if binary_array[i] == 1:
                if current_length == 0:
                    current_start = i
                current_length += 1
            else:
                if current_length >= min_len:
                    if max_len is None or current_length <= max_len:
                        intervals.append((current_start, i - 1))
                current_length = 0
        if current_length >= min_len:
            if max_len is None or current_length <= max_len:
                intervals.append((current_start, len(binary_array) - 1))
        return intervals

    @staticmethod
    def find_column_clusters(H, p=2, b=2.0, min_len=5, max_len=None):
        n, K = H.shape
        L = np.zeros_like(H, dtype=np.int32)
        clusters = []

        # 获取全局因子的最大激活强度
        global_max = np.max(H)

        for k in range(K):
            col = H[:, k]
            col_max = np.max(col)

            # 【核心改进 1】：摒弃基于全局总能量均值的粗暴过滤，防止大 TAD 通道压制小 TAD 导致漏检。
            # 改为基于最大响应值的相对比例过滤（仅过滤几乎无响应的无用随机通道）。
            if col_max < 1e-4 or col_max < (global_max * 0.08):
                continue

            threshold = col_max / b if b != 0 else 0
            L[:, k] = (col >= threshold).astype(np.int32)

            intervals = ClusterUtils.find_all_consecutive_ones(
                L[:, k],
                min_len=min_len,
                max_len=max_len
            )
            clusters.extend(intervals)
        return clusters, L


def rescue_unassigned_gaps(clusters, n, A_sub, boundary_probs, min_len, max_len, density_threshold=1.15):
    """
    【核心新增】：基于 GCN 边界预测和局部区域接触密度的“盲区救援算法”。
    旨在定位对角线上没有被任何 TAD 覆盖的 Gap 区域，
    若该区域内相互作用强度显著高于其与外界的交叉强度，或两端具有较强的 GCN 边界置信度，则强行收复为新的 TAD。
    """
    if not clusters:
        # 如果当前层没有检测到任何域，以整个区间作为候选
        sorted_clusters = []
    else:
        sorted_clusters = sorted(list(set(clusters)), key=lambda x: x[0])

    gaps = []

    # 1. 搜寻所有未分配的 Gap 区间
    if not sorted_clusters:
        gaps.append((0, n - 1))
    else:
        if sorted_clusters[0][0] >= min_len:
            gaps.append((0, sorted_clusters[0][0] - 1))
        for i in range(len(sorted_clusters) - 1):
            curr_e = sorted_clusters[i][1]
            next_s = sorted_clusters[i + 1][0]
            if next_s - curr_e - 1 >= min_len:
                gaps.append((curr_e + 1, next_s - 1))
        if n - 1 - sorted_clusters[-1][1] >= min_len:
            gaps.append((sorted_clusters[-1][1] + 1, n - 1))

    rescued_clusters = list(sorted_clusters)

    # 2. 评估 Gap，执行救援
    for gap_s, gap_e in gaps:
        gap_len = gap_e - gap_s + 1
        if gap_len < min_len:
            continue

        # 计算 Gap 内部的平均交互密度
        gap_matrix = A_sub[gap_s:gap_e + 1, gap_s:gap_e + 1]
        intra_density = np.mean(gap_matrix)

        # 计算 Gap 与左右邻域的外界交互密度
        context = min(gap_len, 8)
        left_bg = A_sub[max(0, gap_s - context):gap_s, gap_s:gap_e + 1] if gap_s > 0 else np.array([])
        right_bg = A_sub[gap_s:gap_e + 1, gap_e + 1:min(n, gap_e + 1 + context)] if gap_e < n - 1 else np.array([])

        bg_vals = []
        if left_bg.size > 0: bg_vals.append(np.mean(left_bg))
        if right_bg.size > 0: bg_vals.append(np.mean(right_bg))

        inter_density = np.mean(bg_vals) if bg_vals else 1e-5
        contrast_ratio = intra_density / (inter_density + 1e-9)

        # 获取 GCN 端点边界预测概率最大值
        edge_w = min(3, gap_len // 2)
        left_prob = np.max(boundary_probs[max(0, gap_s - 1):gap_s + edge_w]) if gap_s > 0 else 0.5
        right_prob = np.max(boundary_probs[gap_e - edge_w:min(n, gap_e + 2)]) if gap_e < n - 1 else 0.5

        # 救援条件：内部凝聚力高（对角线接触多），或者两端预测的边界信号很强
        if contrast_ratio > density_threshold or (left_prob + right_prob > 0.65):
            if gap_len <= max_len:
                rescued_clusters.append((gap_s, gap_e))
            else:
                # 超过 max_len 则利用 GCN 边界概率寻找最佳分割点
                mid_s = gap_s + min_len
                mid_e = gap_e - min_len
                if mid_e > mid_s:
                    split_idx = mid_s + np.argmax(boundary_probs[mid_s:mid_e + 1])
                    if (split_idx - gap_s + 1) >= min_len and (gap_e - split_idx) >= min_len:
                        rescued_clusters.append((gap_s, split_idx))
                        rescued_clusters.append((split_idx + 1, gap_e))
                    else:
                        rescued_clusters.append((gap_s, gap_e))
                else:
                    rescued_clusters.append((gap_s, gap_e))

    return sorted(rescued_clusters, key=lambda x: x[0])


def close_small_gaps_and_expand(clusters, n, max_gap=4):
    if not clusters:
        return []

    sorted_clusters = sorted(list(set(clusters)), key=lambda x: x[0])

    if sorted_clusters[0][0] <= max_gap:
        sorted_clusters[0] = (0, sorted_clusters[0][1])
    if n - 1 - sorted_clusters[-1][1] <= max_gap:
        sorted_clusters[-1] = (sorted_clusters[-1][0], n - 1)

    for i in range(len(sorted_clusters) - 1):
        curr_s, curr_e = sorted_clusters[i]
        next_s, next_e = sorted_clusters[i + 1]

        gap = next_s - curr_e - 1
        if 0 < gap <= max_gap:
            mid = curr_e + (gap // 2) + 1
            sorted_clusters[i] = (curr_s, mid - 1)
            sorted_clusters[i + 1] = (mid, next_e)

    return sorted_clusters


def merge_overlapping_tads(clusters, overlap_threshold=0.7):
    if not clusters:
        return []

    unique_clusters = list(set(clusters))
    unique_clusters.sort(key=lambda x: (x[0], x[1] - x[0]))

    merged = []
    for s_new, e_new in unique_clusters:
        if s_new >= e_new:
            continue

        len_new = e_new - s_new + 1
        is_duplicate = False

        for i, (s_old, e_old) in enumerate(merged):
            intersection = max(0, min(e_new, e_old) - max(s_new, s_old) + 1)
            len_old = e_old - s_old + 1

            overlap_new = intersection / len_new
            overlap_old = intersection / len_old

            if overlap_new >= overlap_threshold and overlap_old >= overlap_threshold:
                is_duplicate = True
                if len_new < len_old:
                    merged[i] = (s_new, e_new)
                break

        if not is_duplicate:
            merged.append((s_new, e_new))

    merged.sort(key=lambda x: x[0])
    return merged


def pre_process(mat, percent=99.9):
    np.nan_to_num(mat, copy=False)
    try:
        mat, _ = KRnorm.KRnorm_sym(mat)
    except:
        pass

    mat = np.log1p(mat)
    if percent < 100:
        p_val = np.percentile(mat, percent)
        mat[mat > p_val] = p_val

    np.fill_diagonal(mat, 0)
    mat = mat / (np.max(mat) + 1e-9)
    return mat


# ==========================================
# 2. AI 模型架构 (安全无阻断版)
# ==========================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.residual = residual
        self.residual_transform = nn.Linear(in_features,
                                            out_features) if residual and in_features != out_features else None

    def forward(self, x, adj):
        B, n, _ = adj.shape
        I = torch.eye(n, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        adj_with_self = adj + I

        degree = torch.sum(adj_with_self, dim=2, keepdim=True) + 1e-5
        d_inv_sqrt = torch.pow(degree, -0.5)
        adj_norm = d_inv_sqrt * adj_with_self * d_inv_sqrt.transpose(1, 2)
        support = self.linear(x)
        output = torch.bmm(adj_norm, support)

        if self.residual:
            residual = self.residual_transform(x) if self.residual_transform else x
            output = output + residual

        return F.relu(output)


class DifferentiableSymINNMF(nn.Module):
    def __init__(self, eps=1e-10, pos_weight=5.0):
        super(DifferentiableSymINNMF, self).__init__()
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, A, X_prime, Y_prime, alpha, lam):
        B, n, K = X_prime.shape
        A_norm = A

        X = F.softmax(X_prime, dim=1)
        Y = F.softmax(Y_prime, dim=1)

        pX = torch.cumsum(X, dim=1)
        pY_shifted = torch.cat([torch.zeros(B, 1, K, device=A.device), torch.cumsum(Y[:, :-1, :], dim=1)], dim=1)
        Z = pX - pY_shifted

        Z_alpha = torch.zeros((B, n, n), device=A.device)
        for k in range(K):
            Z_k = Z[:, :, k].unsqueeze(2)
            alpha_k = alpha[:, k].unsqueeze(1).unsqueeze(2)
            Z_alpha = Z_alpha + alpha_k * torch.bmm(Z_k, Z_k.transpose(1, 2))

        E = A_norm - Z_alpha

        eye = torch.eye(n, device=A.device).unsqueeze(0).expand(B, -1, -1)
        mask = 1.0 - eye
        weight_matrix = torch.where(A > 1e-5, self.pos_weight, 1.0)
        final_weight = mask * weight_matrix

        obj_E = torch.mean(final_weight * (E ** 2))
        obj_X = torch.mean(X * torch.log(X + self.eps))
        obj_Y = torch.mean(Y * torch.log(Y + self.eps))

        mean_lam = torch.mean(lam)
        OBJ = obj_E + mean_lam * (obj_X + obj_Y)

        alpha_expanded = alpha.unsqueeze(1)
        H_new = torch.abs(torch.sqrt(torch.abs(alpha_expanded) + 1e-8) * Z)

        return OBJ, H_new


class DynamicAI_SymNMF_Predictor(nn.Module):
    def __init__(self, max_K=10, hidden_dim=64, dropout_rate=0.3):
        super(DynamicAI_SymNMF_Predictor, self).__init__()
        self.max_K = max_K

        self.gcn1 = GraphConvolution(1, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.predict_X = nn.Linear(hidden_dim, max_K)
        self.predict_Y = nn.Linear(hidden_dim, max_K)
        self.predict_alpha = nn.Linear(hidden_dim, max_K)
        self.predict_lam = nn.Linear(hidden_dim, 1)
        self.predict_boundary = nn.Linear(hidden_dim, 1)
        self.differentiable_nmf = DifferentiableSymINNMF()

    def forward(self, A_sub):
        original_dim = A_sub.dim()
        if original_dim == 2: A_sub = A_sub.unsqueeze(0)
        B, n, _ = A_sub.shape

        node_features = torch.log1p(torch.sum(A_sub, dim=2, keepdim=True))
        eps = 1e-8
        mean = node_features.mean(dim=1, keepdim=True)
        std = node_features.std(dim=1, keepdim=True)
        node_features = (node_features - mean) / (std + eps)

        h1 = self.dropout1(F.relu(self.bn1(self.gcn1(node_features, A_sub).transpose(1, 2)).transpose(1, 2)))
        h2 = self.dropout2(F.relu(self.bn2(self.gcn2(h1, A_sub).transpose(1, 2)).transpose(1, 2)))

        X_prime = self.predict_X(h2)
        Y_prime = self.predict_Y(h2)
        h_global = h2.mean(dim=1)

        alpha = F.softplus(self.predict_alpha(h_global))
        lam = -torch.abs(self.predict_lam(h_global)) * 20.0 - 1.0

        loss_NMF, H_new = self.differentiable_nmf(A_sub, X_prime, Y_prime, alpha, lam)
        boundary_logits = self.predict_boundary(h2).squeeze(-1)

        if original_dim == 2:
            H_new = H_new.squeeze(0)
            boundary_logits = boundary_logits.squeeze(0)

        return loss_NMF, H_new, boundary_logits


# ==========================================
# 3. 预训练流程
# ==========================================
def bounds_to_target_matrix_overlapping(n, bounds, sigma=1.2):
    target = np.zeros((n, n), dtype=np.float32)
    boundary_labels = np.zeros(n, dtype=np.float32)

    for start, end in bounds:
        if start == -1 or end == -1: continue
        target[start:end + 1, start:end + 1] += 1.0

        for center in [start, end]:
            for offset in range(-3, 4):
                pos = center + offset
                if 0 <= pos < n:
                    val = np.exp(-(offset ** 2) / (2 * (sigma ** 2)))
                    boundary_labels[pos] = max(boundary_labels[pos], val)

    target = np.clip(target, 0.0, 1.0)
    return target, boundary_labels


def parse_bounds_file(filepath):
    bounds = []
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return bounds

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    start = int(parts[1]) - 1
                    end = int(parts[5]) - 1
                    bounds.append((start, end))
                except ValueError:
                    continue
    return bounds


class SupervisedSimulatedHiCDataset(Dataset):
    def __init__(self, sim_matrices, sim_bounds_list, samples_per_tad=5, random_bg_samples=20):
        self.samples = []
        for mat, bounds in zip(sim_matrices, sim_bounds_list):
            n = mat.shape[0]
            mat_target, boundary_labels = bounds_to_target_matrix_overlapping(n, bounds)

            for start, end in bounds:
                if start == -1 or end == -1:
                    continue
                tad_size = end - start + 1

                for _ in range(samples_per_tad):
                    context = np.random.randint(10, 30)
                    crop_size = min(max(tad_size + context * 2, 30), n)

                    max_shift = max(0, context - 5)
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    center = (start + end) // 2 + shift

                    half = crop_size // 2
                    c_start = max(0, min(center - half, n - crop_size))

                    sub_mat = mat[c_start:c_start + crop_size, c_start:c_start + crop_size]
                    sub_target = mat_target[c_start:c_start + crop_size, c_start:c_start + crop_size]
                    sub_bounds = boundary_labels[c_start:c_start + crop_size]

                    self.samples.append({
                        'matrix': torch.FloatTensor(sub_mat),
                        'target': torch.FloatTensor(sub_target),
                        'bounds': torch.FloatTensor(sub_bounds)
                    })

            for _ in range(random_bg_samples):
                crop_size = np.random.randint(30, min(100, n + 1))
                c_start = np.random.randint(0, n - crop_size + 1)

                sub_mat = mat[c_start:c_start + crop_size, c_start:c_start + crop_size]
                sub_target = mat_target[c_start:c_start + crop_size, c_start:c_start + crop_size]
                sub_bounds = boundary_labels[c_start:c_start + crop_size]

                self.samples.append({
                    'matrix': torch.FloatTensor(sub_mat),
                    'target': torch.FloatTensor(sub_target),
                    'bounds': torch.FloatTensor(sub_bounds)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_pad_collate(batch):
    max_size = max([item['matrix'].shape[0] for item in batch])
    batched_matrix, batched_target, batched_bounds = [], [], []

    for item in batch:
        mat, tgt, bnd = item['matrix'], item['target'], item['bounds']
        pad_size = max_size - mat.shape[0]

        mat_padded = F.pad(mat, (0, pad_size, 0, pad_size), mode='constant', value=0.0)
        tgt_padded = F.pad(tgt, (0, pad_size, 0, pad_size), mode='constant', value=0.0)
        bnd_padded = F.pad(bnd, (0, pad_size), mode='constant', value=0.0)

        batched_matrix.append(mat_padded)
        batched_target.append(tgt_padded)
        batched_bounds.append(bnd_padded)

    return {
        'matrix': torch.stack(batched_matrix, dim=0),
        'target': torch.stack(batched_target, dim=0),
        'bounds': torch.stack(batched_bounds, dim=0)
    }


def differentiable_soft_iou_loss(pred_logits, target_labels, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = torch.sum(pred * target_labels, dim=-1)
    union = torch.sum(pred, dim=-1) + torch.sum(target_labels, dim=-1) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - torch.mean(iou)


def intra_inter_contrast_loss(H_recon, M_target, eps=1e-6):
    intra_mask = M_target
    inter_mask = 1.0 - M_target

    intra_mean = torch.sum(H_recon * intra_mask, dim=(1, 2)) / (torch.sum(intra_mask, dim=(1, 2)) + eps)
    inter_mean = torch.sum(H_recon * inter_mask, dim=(1, 2)) / (torch.sum(inter_mask, dim=(1, 2)) + eps)

    contrast = inter_mean - intra_mean
    return torch.mean(contrast)


def supervised_pretrain(model, dataloader, val_dataloader=None, epochs=5, lr=1e-3,
                        gamma_mse=1.0, gamma_bce=3.0, gamma_iou=2.0, gamma_contrast=0.5, gamma_tv=0.8,
                        device='cuda', patience=3, min_delta=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            A_sub = batch['matrix'].to(device)
            M_target = batch['target'].to(device)
            B_target = batch['bounds'].to(device)

            optimizer.zero_grad()
            loss_NMF, H_new, boundary_logits = model(A_sub)
            H_recon = torch.bmm(H_new, H_new.transpose(1, 2))

            loss_mse = mse(H_recon, M_target)
            loss_contrast = intra_inter_contrast_loss(H_recon, M_target)
            loss_bce = bce_loss(boundary_logits, B_target)
            loss_iou = differentiable_soft_iou_loss(boundary_logits, B_target)
            loss_tv = torch.mean(torch.abs(H_new[:, 1:, :] - H_new[:, :-1, :]))

            loss = loss_NMF + gamma_mse * loss_mse + gamma_bce * loss_bce + gamma_iou * loss_iou + gamma_contrast * loss_contrast + gamma_tv * loss_tv

            epoch_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)

        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    A_sub_val = batch['matrix'].to(device)
                    M_target_val = batch['target'].to(device)
                    B_target_val = batch['bounds'].to(device)

                    loss_NMF_val, H_new_val, boundary_logits_val = model(A_sub_val)
                    H_recon_val = torch.bmm(H_new_val, H_new_val.transpose(1, 2))

                    loss_mse_val = mse(H_recon_val, M_target_val)
                    loss_contrast_val = intra_inter_contrast_loss(H_recon_val, M_target_val)
                    loss_bce_val = bce_loss(boundary_logits_val, B_target_val)
                    loss_iou_val = differentiable_soft_iou_loss(boundary_logits_val, B_target_val)
                    loss_tv_val = torch.mean(torch.abs(H_new_val[:, 1:, :] - H_new_val[:, :-1, :]))

                    loss_val = loss_NMF_val + gamma_mse * loss_mse_val + gamma_bce * loss_bce_val + gamma_iou * loss_iou_val + gamma_contrast * loss_contrast_val + gamma_tv * loss_tv_val
                    val_loss += loss_val.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

            if early_stopping.step(avg_val_loss, model):
                print(f"[Pre-training Early Stopped] Loading previously best profiles...")
                model.load_state_dict(early_stopping.best_state)
                break
        else:
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
            if early_stopping.step(avg_loss, model):
                print(f"[Pre-training 早停止] 加载历史最佳权重中...")
                model.load_state_dict(early_stopping.best_state)
                break

    if not early_stopping.early_stop and early_stopping.best_state is not None:
        print("[Pre-training 结束] 加载整个预训练过程中指标最佳的模型参数。")
        model.load_state_dict(early_stopping.best_state)

    return model


# ==========================================
# 4. 局部微调与后处理
# ==========================================
def optimize_params_with_AI(A_sub_np, model, base_params, fine_tune_epochs=20, device='cuda'):
    n = A_sub_np.shape[0]
    if n < base_params['min_tad_bins']: return []

    local_model = copy.deepcopy(model).to(device)

    A_tensor = torch.FloatTensor(A_sub_np).to(device)
    if A_tensor.dim() == 2:
        A_tensor = A_tensor.unsqueeze(0)

    if fine_tune_epochs > 0:
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-4)
        local_model.train()
        freeze_bn(local_model)
        for _ in range(fine_tune_epochs):
            optimizer.zero_grad()
            loss_NMF, H_new, _ = local_model(A_tensor)

            if H_new.dim() == 3:
                tv_penalty = torch.mean(torch.abs(H_new[:, 1:, :] - H_new[:, :-1, :]))
            else:
                tv_penalty = torch.mean(torch.abs(H_new[1:, :] - H_new[:-1, :]))

            total_loss = loss_NMF + 5e-3 * tv_penalty
            total_loss.backward()
            optimizer.step()

    local_model.eval()
    with torch.no_grad():
        _, H_new, boundary_logits = local_model(A_tensor)

    if H_new.dim() == 3:
        H_new = H_new.squeeze(0)
        boundary_logits = boundary_logits.squeeze(0)

    H_np = H_new.cpu().numpy()
    boundary_probs = torch.sigmoid(boundary_logits).cpu().numpy()  # 获取预测边界概率

    if np.isnan(H_np).any(): return []

    min_len = base_params['min_tad_bins']
    max_len = min(base_params['max_tad_bins'], n)

    # 1. 提取聚类候选区间 (经过能量过滤优化)
    clusters, _ = ClusterUtils.find_column_clusters(
        H_np,
        p=base_params['p'],
        b=base_params['b'],
        min_len=min_len,
        max_len=max_len
    )

    valid_clusters = [(s, e) for s, e in clusters if s != -1 and not (s == 0 and e == n - 1)]

    # 2. 重叠去重
    overlap_threshold = base_params.get('overlap_threshold', 0.7)
    final_clusters = merge_overlapping_tads(valid_clusters, overlap_threshold=overlap_threshold)

    # 3. 【核心改进 2】：盲区 Gap 双重机制恢复，强力解决空档不识别的问题
    final_clusters = rescue_unassigned_gaps(
        final_clusters,
        n,
        A_sub_np,
        boundary_probs,
        min_len=min_len,
        max_len=max_len,
        density_threshold=1.45
    )

    # 4. 微小空隙闭合与扩张
    max_gap = base_params.get('max_gap_to_close', 4)
    final_clusters = close_small_gaps_and_expand(final_clusters, n, max_gap=max_gap)

    return final_clusters


# ==========================================
# 5. 主干网络: 层次化 TAD 识别
# ==========================================
def main_hierarchical(A, model, datafile_name, max_layers=2, p=5, b=2.0, overlap_threshold=0.7, device='cuda',
                      write_tsv="./output_tsv",
                      data_stat="real", resolution=40000):
    n = A.shape[0]

    params_env = Parameters(resolution=resolution, b=b)
    base_params = {
        'p': p,
        'b': params_env.b,
        'overlap_threshold': overlap_threshold,
        'min_tad_bins': params_env.min_tad_bins,
        'max_tad_bins': params_env.max_tad_bins,
        'max_gap_to_close': params_env.max_gap_to_close
    }

    hierarchical_tads = []
    current_regions = [(0, n - 1)]

    for layer in range(max_layers):
        print(f"--- Processing Layer {layer + 1} ---")
        current_layer_tads = []

        for global_start, global_end in current_regions:
            region_size = global_end - global_start + 1

            if region_size < params_env.min_parent_to_split:
                current_layer_tads.append((global_start, global_end))
                continue

            sub_A = A[global_start:global_end + 1, global_start:global_end + 1]
            local_clusters = optimize_params_with_AI(sub_A, model, base_params, fine_tune_epochs=0, device=device)

            if not local_clusters:
                current_layer_tads.append((global_start, global_end))
                continue

            for local_start, local_end in local_clusters:
                new_start = global_start + local_start
                new_end = global_start + local_end

                if (new_end - new_start + 1) >= params_env.min_tad_bins:
                    current_layer_tads.append((new_start, new_end))
                    print(f"Layer {layer + 1} Found TAD: {new_start} to {new_end} (Size: {new_end - new_start + 1})")

        if not current_layer_tads:
            break

        current_layer_tads = merge_overlapping_tads(current_layer_tads, overlap_threshold=overlap_threshold)
        current_layer_tads = close_small_gaps_and_expand(current_layer_tads, n, max_gap=params_env.max_gap_to_close)

        hierarchical_tads.append(current_layer_tads)
        current_regions = current_layer_tads

    write2file(save_tsv=write_tsv, hierarchical_tads=hierarchical_tads, datafile_name=datafile_name,
               data_stat=data_stat)
    return hierarchical_tads


def write2file(hierarchical_tads, datafile_name, save_tsv, data_stat="real"):
    if not os.path.exists(save_tsv):
        os.makedirs(save_tsv)

    file_name = os.path.join(save_tsv, datafile_name + ".tsv")
    output = []

    if data_stat == "sim":
        chr_name = "sim"
        resolution = 40000
    else:
        chr_start = datafile_name.find("chr")
        chr_pos = datafile_name.find("_KR")
        chr_name = datafile_name[chr_start:chr_pos] if chr_start != -1 and chr_pos != -1 else "chr_unknown"
        match = re.search(r'(\d+)kb', datafile_name)
        resolution = int(match.group(1)) * 1000 if match else 40000

    for layer_idx, layer_tads in enumerate(hierarchical_tads):
        for start, end in layer_tads:
            bin_start = start + 1
            bin_end = end + 1
            line = [
                chr_name,
                bin_start, (bin_start - 1) * resolution, bin_start * resolution,
                chr_name,
                bin_end, (bin_end - 1) * resolution, bin_end * resolution
            ]
            output.append(line)

    output_df = pd.DataFrame(output)
    output_df.to_csv(file_name, sep='\t', index=False, header=False)
    print("write boundaries successfully: " + file_name)


def load_matrix(filepath):
    mat = np.loadtxt(filepath, dtype=np.float32)
    mat = pre_process(mat)
    return mat


def load_pretrain_paths_from_txt(txt_file_path):
    matrix_dirs = []
    bound_dirs = []

    if not os.path.exists(txt_file_path):
        print(f"Error: Path file not found: {txt_file_path}")
        return matrix_dirs, bound_dirs

    with open(txt_file_path, 'r') as f:
        for line in f:
            root_dir = line.strip()
            if not root_dir:
                continue

            matrix_dir = os.path.join(root_dir, "matrix")
            bound_dir = os.path.join(root_dir, "gt")

            if os.path.exists(matrix_dir):
                matrix_dirs.append(matrix_dir)
            else:
                print(f"Warning: Matrix directory not found: {matrix_dir}")

            if os.path.exists(bound_dir):
                bound_dirs.append(bound_dir)
            else:
                print(f"Warning: Bound directory not found: {bound_dir}")

    return matrix_dirs, bound_dirs


def load_paired_pretrain_data_multiple(matrix_dirs, bound_dirs):
    all_sim_matrices = []
    all_sim_bounds_list = []

    if isinstance(matrix_dirs, str): matrix_dirs = [matrix_dirs]
    if isinstance(bound_dirs, str): bound_dirs = [bound_dirs]

    if len(matrix_dirs) != len(bound_dirs):
        raise ValueError("Matrix directories and Bound directories must have the same length.")

    for m_dir, b_dir in zip(matrix_dirs, bound_dirs):
        print(f"Searching in: Matrix Dir: {m_dir} | Bound Dir: {b_dir}")

        if not os.path.exists(m_dir) or not os.path.exists(b_dir):
            print(f"Warning: One of the paths does not exist, skipping: {m_dir} or {b_dir}")
            continue

        for filename in os.listdir(m_dir):
            if filename.endswith(".txt"):
                base_name = os.path.splitext(filename)[0]
                matrix_path = os.path.join(m_dir, filename)
                bound_path = os.path.join(b_dir, f"{base_name}.domains")

                if os.path.exists(bound_path):
                    try:
                        mat = load_matrix(matrix_path)
                        bounds = parse_bounds_file(bound_path)

                        all_sim_matrices.append(mat)
                        all_sim_bounds_list.append(bounds)
                    except Exception as e:
                        print(f"  [Error] Failed to load {filename}: {e}")
                else:
                    print(f"  [Skip] No matching bound for {filename}")

    return all_sim_matrices, all_sim_bounds_list


def load_matrix_dirs_from_txt(txt_file_path):
    matrix_dirs = []
    if not os.path.exists(txt_file_path):
        print(f"Error: Path file not found: {txt_file_path}")
        return matrix_dirs

    with open(txt_file_path, 'r') as f:
        for line in f:
            root_dir = line.strip()
            if not root_dir:
                continue

            matrix_dir = os.path.join(root_dir, "matrix")
            if os.path.exists(matrix_dir):
                matrix_dirs.append(matrix_dir)
            else:
                if os.path.exists(root_dir):
                    matrix_dirs.append(root_dir)
                else:
                    print(f"Warning: Directory not found: {root_dir}")
    return matrix_dirs


def load_matrix_dirs_from_txt2(txt_file_path):
    dirs = []
    if not os.path.exists(txt_file_path):
        print(f"Error: Path file not found: {txt_file_path}")
        return dirs

    with open(txt_file_path, 'r') as f:
        for line in f:
            root_dir = line.strip()
            if not root_dir:
                continue

            if os.path.exists(root_dir):
                dirs.append(root_dir)
            else:
                print(f"Warning: Directory not found: {root_dir}")
    return dirs


def load_matrices_from_dirs(matrix_dirs):
    matrices = []
    filenames = []

    for m_dir in matrix_dirs:
        if not os.path.exists(m_dir):
            print(f"Warning: Directory not found: {m_dir}")
            continue

        for filename in sorted(os.listdir(m_dir)):
            f_path = os.path.join(m_dir, filename)

            if filename.endswith(".txt"):
                try:
                    mat = load_matrix(f_path)
                    matrices.append(mat)
                    filenames.append(filename)
                except Exception as e:
                    print(f"  [Error] Failed to load {filename}: {e}")

            elif filename.endswith(".npz"):
                try:
                    mat = load_npz(f_path)
                    mat = mat.toarray()
                    mat = pre_process(mat)
                    matrices.append(mat)
                    filenames.append(filename)
                except Exception as e:
                    print(f"  [Error] Failed to load {filename}: {e}")

    return matrices, filenames


def load_matrices_from_dirs2(file_list_path):
    matrices = []
    filenames = []

    if not os.path.exists(file_list_path):
        print(f"Warning: File list not found: {file_list_path}")
        return matrices, filenames

    with open(file_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for path in lines:
        if not os.path.exists(path):
            print(f"Warning: Path not found: {path}")
            continue

        if os.path.isdir(path):
            for filename in sorted(os.listdir(path)):
                f_path = os.path.join(path, filename)
                if not os.path.isfile(f_path):
                    continue

                if filename.endswith(".txt"):
                    try:
                        mat = load_matrix(f_path)
                        matrices.append(mat)
                        filenames.append(filename)
                    except Exception as e:
                        print(f"  [Error] Failed to load {filename}: {e}")

                elif filename.endswith(".npz"):
                    try:
                        mat = load_npz(f_path)
                        mat = mat.toarray()
                        mat = pre_process(mat)
                        matrices.append(mat)
                        filenames.append(filename)
                    except Exception as e:
                        print(f"  [Error] Failed to load {filename}: {e}")

        elif os.path.isfile(path):
            filename = os.path.basename(path)

            if filename.endswith(".txt"):
                try:
                    mat = load_matrix(path)
                    matrices.append(mat)
                    filenames.append(filename)
                except Exception as e:
                    print(f"  [Error] Failed to load {filename}: {e}")

            elif filename.endswith(".npz"):
                try:
                    mat = load_npz(path)
                    mat = mat.toarray()
                    matrices.append(mat)
                    filenames.append(filename)
                except Exception as e:
                    print(f"  [Error] Failed to load {filename}: {e}")

    return matrices, filenames


# ==========================================
# 6. 全局微调 (含早停止逻辑)
# ==========================================
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def global_fine_tune(model, target_matrices, epochs=10, lr=1e-5, device='cuda', crop_size=100, reg_weight=10.0,
                     patience=5, min_delta=1e-5):
    model.train()
    freeze_bn(model)

    pretrained_model = copy.deepcopy(model)
    pretrained_model.eval()
    for param in pretrained_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Preparing static dataset for fine-tuning...")
    static_sub_matrices = []
    for mat in target_matrices:
        n = mat.shape[0]
        if n < crop_size:
            sub_mat = mat
        else:
            start = np.random.randint(0, n - (crop_size - 1))
            sub_mat = mat[start:start + crop_size, start:start + crop_size]
        static_sub_matrices.append(torch.FloatTensor(sub_mat).unsqueeze(0).to(device))

    print(f"Starting global adaptation with Feature-level Anchoring...")

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)

    for epoch in range(epochs):
        total_loss = 0
        total_nmf_loss = 0
        total_reg_loss = 0

        model.train()
        freeze_bn(model)

        for sub_A in static_sub_matrices:
            optimizer.zero_grad()

            loss_NMF, H_new, _ = model(sub_A)

            with torch.no_grad():
                _, H_ref, _ = pretrained_model(sub_A)

            loss_reg = torch.mean((H_new - H_ref) ** 2)
            loss = loss_NMF + reg_weight * loss_reg

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_nmf_loss += loss_NMF.item()
            total_reg_loss += loss_reg.item()

        avg_loss = total_loss / len(static_sub_matrices)
        avg_nmf = total_nmf_loss / len(static_sub_matrices)
        avg_reg = total_reg_loss / len(static_sub_matrices)

        weighted_reg_contribution = reg_weight * avg_reg
        print(f"Global Fine-tune Epoch {epoch + 1:2d}/{epochs} | "
              f"Total Loss: {avg_loss:.4f} | "
              f"NMF Loss: {avg_nmf:.4f} | "
              f"Weighted Reg: {weighted_reg_contribution:.4f} (Raw Reg: {avg_reg:.6f})")

        if early_stopping.step(avg_loss, model):
            print(f"[Global Fine-tuning 早停止] 提前结束。加载历史最佳权重中...")
            model.load_state_dict(early_stopping.best_state)
            break

    if not early_stopping.early_stop and early_stopping.best_state is not None:
        print("[Global Fine-tuning 结束] 加载整个微调过程中指标最佳的模型参数。")
        model.load_state_dict(early_stopping.best_state)

    return model


def plot_hierarchical_tad_heatmap(A, hierarchical_tads, title="Hierarchical TAD Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 8))

    cax = ax.imshow(A, cmap='Reds', interpolation='none', origin='upper')
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']

    for layer_idx, layer_tads in enumerate(hierarchical_tads):
        color = colors[layer_idx % len(colors)]
        linestyle = linestyles[layer_idx % len(linestyles)]

        for k, (start, end) in enumerate(layer_tads):
            length = end - start + 1
            rect = patches.Rectangle(
                (start - 0.5, start - 0.5),
                length,
                length,
                linewidth=2,
                edgecolor=color,
                linestyle=linestyle,
                facecolor='none',
                label=f'Layer {layer_idx + 1}' if k == 0 else ""
            )
            ax.add_patch(rect)

    plt.title(title)
    plt.xlabel('Bin Index')
    plt.ylabel('Bin Index')
    plt.legend(loc='upper right')
    plt.show()


# ==========================================
# 7. 主运行程序
# ==========================================
if __name__ == "__main__":
    profile = "hpc"
    write_tsv = "/gpfs1/home/qiusliang2/result8/hic/v7_182/"
    # profile = "autoDL"
    # write_tsv = "/root/autodl-tmp/results/hic/v7_182/"
    data_stat = "sim"
    plot_name = ""

    max_K = 30
    resolution = 25000

    if profile == "hpc":
        home_root = "/gpfs1/home/qiusliang2/"
        data_storage_root = "/gpfs1/home/qiusliang2/data_storage/synmf/"
    elif profile == "autoDL":
        home_root = "/root/"
        data_storage_root = "/root/autodl-tmp/"
    else:
        print("ERROR! Wrong profile input")
        exit(0)

    PRETRAIN_PATHS_TXT = home_root + "synmf/pretrain.txt"
    VAL_PATHS_TXT = home_root + "synmf/valid.txt"
    FINETUNE_PATHS_TXT = home_root + "synmf/finetune.txt"
    TEST_PATHS_TXT = home_root + "synmf/test.txt"

    MODEL_SAVE_PATH = data_storage_root + "trained_synmf_modelv7_182_K30.pth"
    MODEL_LOAD_PATH = data_storage_root + "trained_synmf_modelv7_182_K30.pth"
    MODEL_LOAD_PATH = ""  # 第一次训练时留空
    ADAPT_MODEL_SAVE_PATH = data_storage_root + "adapted_modelv7_182_K30.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    if MODEL_LOAD_PATH:
        model = DynamicAI_SymNMF_Predictor(max_K=max_K)
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Loaded model from {MODEL_LOAD_PATH}")
    else:
        print("Loading pre-training paths from txt file...")
        PRETRAIN_MATRIX_DIRS, PRETRAIN_BOUND_DIRS = load_pretrain_paths_from_txt(PRETRAIN_PATHS_TXT)

        print("Loading validation paths from txt file...")
        VAL_MATRIX_DIRS, VAL_BOUND_DIRS = load_pretrain_paths_from_txt(VAL_PATHS_TXT)

        if not PRETRAIN_MATRIX_DIRS:
            print("Error: No valid pre-training directories found. Please check your txt file.")
            exit(1)

        print(f"Found {len(PRETRAIN_MATRIX_DIRS)} pre-training directories")
        if VAL_MATRIX_DIRS:
            print(f"Found {len(VAL_MATRIX_DIRS)} validation directories")

        print("Loading pre-training data from multiple sources...")
        sim_matrices, sim_bounds_list = load_paired_pretrain_data_multiple(
            PRETRAIN_MATRIX_DIRS,
            PRETRAIN_BOUND_DIRS
        )

        val_matrices = []
        val_bounds_list = []
        if VAL_MATRIX_DIRS:
            print("Loading validation data from multiple sources...")
            val_matrices, val_bounds_list = load_paired_pretrain_data_multiple(
                VAL_MATRIX_DIRS,
                VAL_BOUND_DIRS
            )

        if not sim_matrices:
            print("Error: No pre-training data loaded. Please check your paths.")
            exit(1)

        print("Initialize Model...")
        model = DynamicAI_SymNMF_Predictor(max_K=max_K)
        model = model.to(device)

        dataset = SupervisedSimulatedHiCDataset(
            sim_matrices,
            sim_bounds_list,
            samples_per_tad=10,
            random_bg_samples=20
        )
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=custom_pad_collate, num_workers=4,
                                pin_memory=True)

        val_dataloader = None
        if val_matrices:
            val_dataset = SupervisedSimulatedHiCDataset(
                val_matrices,
                val_bounds_list,
                samples_per_tad=10,
                random_bg_samples=20
            )
            val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=custom_pad_collate,
                                        num_workers=4, pin_memory=True)

        print("Start Pre-training...")
        model = supervised_pretrain(model, dataloader, val_dataloader=val_dataloader, epochs=10, device=device, lr=1e-3,
                                    patience=3, gamma_tv=0.8)

        save_dir = os.path.dirname(MODEL_SAVE_PATH)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model successfully saved to: {MODEL_SAVE_PATH}")

    # ==========================================
    # 2. 全局微调
    # ==========================================
    print("\nLoading all fine-tune target matrices from txt...")
    finetune_matrix_dirs = load_matrix_dirs_from_txt2(FINETUNE_PATHS_TXT)
    if not finetune_matrix_dirs:
        print("Error: No valid fine-tune directories found. Please check your txt file.")
        exit(1)

    target_matrices, _ = load_matrices_from_dirs(finetune_matrix_dirs)

    if not target_matrices:
        print("No target matrices found for fine-tuning. Exiting.")
        exit(1)

    print(f"Loaded {len(target_matrices)} matrices for fine-tuning.")

    model = global_fine_tune(model, target_matrices, epochs=20, lr=1e-5, device=device, crop_size=100, reg_weight=10.0,
                             patience=5)
    torch.save(model.state_dict(), ADAPT_MODEL_SAVE_PATH)
    print(f"Fine-tuned model successfully saved to: {ADAPT_MODEL_SAVE_PATH}")

    # ==========================================
    # 3. 测试阶段
    # ==========================================
    print("\nLoading all test matrices from txt...")
    test_matrices, test_files = load_matrices_from_dirs2(TEST_PATHS_TXT)

    if not test_matrices:
        print("No test matrices found. Exiting.")
        exit(1)

    print(f"Loaded {len(test_matrices)} matrices for testing.")
    print("\nStarting Hierarchical TAD Calling...")
    results = {}

    for filename, mat in zip(test_files, test_matrices):
        print(f"\n>>> Processing: {filename}")
        tads = main_hierarchical(mat, model, max_layers=2, p=1, b=1.5, overlap_threshold=0.55, device=device,
                                 write_tsv=write_tsv,
                                 datafile_name=filename,
                                 data_stat=data_stat, resolution=resolution)
        print("Final Result:", tads)
        results[filename] = tads

        if plot_name:
            plot_hierarchical_tad_heatmap(A=mat, hierarchical_tads=tads, title=plot_name)
