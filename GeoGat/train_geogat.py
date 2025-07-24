
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import numpy as np
import os
import argparse
from torch.utils.data import Dataset
from dataprocess_ply import load_patches_from_h5
from gat import GeoGATModel
import time
from tqdm import tqdm
import glob
import random
from torch_geometric.data import Data, Batch
from torch.amp import autocast, GradScaler

class PatchDataset(Dataset):
    """将列表转换为 Dataset 子类"""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def validate_and_fix_edge_index(data, center):
    """验证并修复 edge_index，确保索引在 [0, num_nodes)"""
    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    if edge_index.size(1) == 0:
        print(f"警告: 补丁 patch_{center} 的 edge_index 为空")
        return data

    max_index = edge_index.max().item()
    min_index = edge_index.min().item()
    if max_index >= num_nodes or min_index < 0:
        print(f"错误: 补丁 patch_{center} 的 edge_index 无效: max={max_index}, min={min_index}, num_nodes={num_nodes}")
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & (edge_index[0] >= 0) & (edge_index[1] >= 0)
        if valid_mask.sum() == 0:
            print(f"警告: 补丁 {center} 无有效边，将作为无边图处理")
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = torch.empty((0, data.edge_attr.size(1) if data.edge_attr.numel() > 0 else 2), dtype=torch.float)
            return data
        valid_edge_index = edge_index[:, valid_mask]
        valid_edge_attr = data.edge_attr[valid_mask]
        print(f"修复: 保留 {valid_mask.sum()} 条有效边，原边数 {edge_index.size(1)}")
        data.edge_index = valid_edge_index
        data.edge_attr = valid_edge_attr
    return data

def validate_batch_edge_index(batch, batch_idx):
    """验证批处理的 edge_index，确保索引在 [0, num_nodes)"""
    num_nodes = batch.x.size(0)
    edge_index = batch.edge_index
    if edge_index.size(1) == 0:
        print(f"警告: Batch {batch_idx} 的 edge_index 为空，补丁中心: {batch.center.tolist()}")
        return batch

    max_index = edge_index.max().item()
    min_index = edge_index.min().item()
    if max_index >= num_nodes or min_index < 0:
        print(f"错误: Batch {batch_idx} 的 edge_index 无效: max={max_index}, min={min_index}, num_nodes={num_nodes}")
        print(f"Batch {batch_idx} 包含补丁中心: {batch.center.tolist()}")
        batch_data = batch.to_data_list()
        valid_data = []
        for i, data in enumerate(batch_data):
            data_num_nodes = data.x.size(0)
            if data.edge_index.size(1) > 0:
                data_max_index = data.edge_index.max().item()
                if data_max_index >= data_num_nodes or data.edge_index.min().item() < 0:
                    print(f"错误: 补丁 {data.center.item()} 的 edge_index 无效: max={data_max_index}, num_nodes={data_num_nodes}")
                    continue
                valid_data.append(data)
            else:
                print(f"警告: 补丁 {data.center.item()} 的 edge_index 为空")
                valid_data.append(data)
        if not valid_data:
            raise ValueError(f"Batch {batch_idx} 中无有效补丁")
        batch = Batch.from_data_list(valid_data)
        print(f"修复: Batch {batch_idx} 保留 {len(valid_data)} 个补丁")
    return batch

def train_gat(train_dataset, val_dataset, test_dataset, num_epochs=100, batch_size=8, model_path="gat_model.pth",
              device="cpu"):
    """训练并测试 GAT 模型，使用默认采样器"""
    model = GeoGATModel(in_channels=6, hidden_channels=128, out_channels=32, heads=4, num_classes=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

    pos_count = sum(1 for data in train_dataset if data.y.item() == 1)
    neg_count = sum(1 for data in train_dataset if data.y.item() == 0)
    pos_weight = neg_count / pos_count if pos_count > 0 else 10.0
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    if device.startswith('cuda'):
        class_weights = class_weights.half()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler('cuda') if device.startswith('cuda') else GradScaler()

    best_val_auc = 0
    best_metrics = {}
    roc_pr_data = {}

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.startswith('cuda')))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.startswith('cuda')))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.startswith('cuda')))

        print(f"Epoch {epoch + 1}: train_loader 批次数: {len(train_loader)}")
        if len(train_loader) == 0:
            raise ValueError("train_loader 为空，请检查数据集")

        model.train()
        total_train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False, mininterval=1.0,
                                 ncols=100)
        for batch_idx, batch in enumerate(train_loader_tqdm):
            start_time = time.time()
            batch = batch.to(device)
            batch = validate_batch_edge_index(batch, batch_idx)  # 验证批处理的 edge_index
            optimizer.zero_grad()
            with autocast(device_type='cuda' if device.startswith('cuda') else 'cpu'):
                out = model(batch)
                loss = criterion(out, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            batch_time = time.time() - start_time
            train_loader_tqdm.set_postfix(batch_time=f"{batch_time:.2f}s")
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_probs = []
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation", leave=False, mininterval=1.0,
                                   ncols=100)
            for batch in val_loader_tqdm:
                batch = batch.to(device)
                batch = validate_batch_edge_index(batch, "val")  # 验证验证集批处理的 edge_index
                with autocast(device_type='cuda' if device.startswith('cuda') else 'cpu'):
                    out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
            val_aupr = average_precision_score(val_labels, val_probs)
            val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)
            val_precision, val_recall, _ = precision_recall_curve(val_labels, val_probs)
        except ValueError as e:
            print(f"错误: 验证集指标计算失败: {e}")
            print(f"验证标签 (前10): {val_labels[:10]}")
            print(f"预测概率 (前10): {val_probs[:10]}")
            continue

        test_loss = 0
        test_correct = 0
        test_total = 0
        test_labels = []
        test_probs = []
        with torch.no_grad():
            test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1} Testing", leave=False, mininterval=1.0,
                                    ncols=100)
            for batch in test_loader_tqdm:
                batch = batch.to(device)
                batch = validate_batch_edge_index(batch, "test")  # 验证测试集批处理的 edge_index
                with autocast(device_type='cuda' if device.startswith('cuda') else 'cpu'):
                    out = model(batch)
                loss = criterion(out, batch.y)
                test_loss += loss.item()
                pred = out.argmax(dim=1)
                test_correct += (pred == batch.y).sum().item()
                test_total += batch.y.size(0)
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_labels.extend(batch.y.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        try:
            test_auc = roc_auc_score(test_labels, test_probs)
            test_aupr = average_precision_score(test_labels, test_probs)
            test_fpr, test_tpr, _ = roc_curve(test_labels, test_probs)
            test_precision, test_recall, _ = precision_recall_curve(test_labels, test_probs)
        except ValueError as e:
            print(f"错误: 测试集指标计算失败: {e}")
            print(f"测试标签 (前10): {test_labels[:10]}")
            print(f"预测概率 (前10): {test_probs[:10]}")
            continue

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {
                'val_auc': val_auc,
                'val_aupr': val_aupr,
                'val_accuracy': val_accuracy,
                'test_auc': test_auc,
                'test_aupr': test_aupr,
                'test_accuracy': test_accuracy
            }
            roc_pr_data = {
                'val_fpr': val_fpr,
                'val_tpr': val_tpr,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_auc': val_auc,
                'val_aupr': val_aupr,
                'test_fpr': test_fpr,
                'test_tpr': test_tpr,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_auc': test_auc,
                'test_aupr': test_aupr
            }
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型到 {model_path}, Best Val AUC: {best_val_auc:.4f}")

    if roc_pr_data:
        np.savez('roc_pr_curve_data.npz', **roc_pr_data)
        print(f"ROC 和 PR 曲线数据保存到 roc_pr_curve_data.npz")
        print(f"最佳性能: Val AUC: {best_metrics['val_auc']:.4f}, Val AUPR: {best_metrics['val_aupr']:.4f}, "
              f"Test AUC: {best_metrics['test_auc']:.4f}, Test AUPR: {best_metrics['test_aupr']:.4f}")

    return model, best_metrics

def main():
    parser = argparse.ArgumentParser(
        description="训练 GAT 模型，评估验证集和测试集的 AUC 和 AUPR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--h5_dir", required=True, help="补丁数据的 HDF5 文件目录")
    parser.add_argument("--model_path", default="gat_model.pth", help="模型保存路径")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu 或 cuda)")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--max_h5_files", type=int, default=None, help="最大加载的 HDF5 文件数量，None 表示加载全部")
    parser.add_argument("--epoch", type=int, default=100, help="训练轮数")
    args = parser.parse_args()

    total_start_time = time.time()

    h5_files = glob.glob(os.path.join(args.h5_dir, "**/patches.h5"), recursive=True)
    if not h5_files:
        raise FileNotFoundError(f"在目录 {args.h5_dir} 中未找到 patches.h5 文件")

    if args.max_h5_files is not None:
        h5_files = random.sample(h5_files, min(args.max_h5_files, len(h5_files)))
        print(f"限制加载 {len(h5_files)} 个 HDF5 文件（总共找到 {len(h5_files)} 个）")
    else:
        print(f"加载全部 {len(h5_files)} 个 HDF5 文件")

    dataset = []
    skipped_patches = []
    for h5_file in h5_files:
        print(f"加载 HDF5 文件: {h5_file}")
        for batch_data in tqdm(load_patches_from_h5(h5_file), desc=f"Loading patches from {h5_file}"):
            for data in batch_data:
                if data is None:
                    print("警告: 遇到了 None 数据，跳过")
                    continue
                if not hasattr(data, 'edge_attr'):
                    print(f"错误: 补丁 patch_{data.center.item()} 缺少 edge_attr，文件: {h5_file}")
                    skipped_patches.append(data.center.item())
                    continue
                center = data.center.item()  # Store center before potential None
                data = validate_and_fix_edge_index(data, center)
                if data is None:
                    print(f"跳过补丁 patch_{center} 由于无效 edge_index")
                    skipped_patches.append(center)
                    continue
                if data.x.size(0) < 1:
                    print(f"错误: 补丁 patch_{center} 数据无效: num_nodes={data.x.size(0)}")
                    skipped_patches.append(center)
                    continue
                dataset.append(data)
        print(f"从 {h5_file} 加载 {len(dataset)} 个有效补丁，跳过 {len(skipped_patches)} 个补丁")

    print(f"总共加载 {len(dataset)} 个有效补丁，跳过 {len(skipped_patches)} 个补丁")
    if skipped_patches:
        print(f"跳过的补丁: {skipped_patches[:10]}")

    if not dataset:
        raise ValueError("数据集为空，请检查 HDF5 文件或修复 edge_index 问题")

    for i, data in enumerate(dataset):
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y') or not hasattr(data, 'edge_attr'):
            print(f"数据集索引 {i} 的样本缺少属性：x={hasattr(data, 'x')}, edge_index={hasattr(data, 'edge_index')}, "
                  f"y={hasattr(data, 'y')}, edge_attr={hasattr(data, 'edge_attr')}")
        if data.y.item() not in [0, 1]:
            print(f"数据集索引 {i} 的样本标签无效：{data.y.item()}")

    pos_count = sum(1 for data in dataset if data.y.item() == 1)
    neg_count = sum(1 for data in dataset if data.y.item() == 0)
    other_count = len(dataset) - pos_count - neg_count
    print(f"数据集正样本: {pos_count}, 负样本: {neg_count}, 其他标签: {other_count}")
    if other_count > 0:
        raise ValueError(f"发现 {other_count} 个无效标签，请检查 load_patches_from_h5")

    dataset = PatchDataset(dataset)
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.25, random_state=42)
    print(f"\n数据集划分: 训练集 {len(train_dataset)} 个补丁, 验证集 {len(val_dataset)} 个补丁, 测试集 {len(test_dataset)} 个补丁")

    pos_count = sum(1 for data in train_dataset if data.y.item() == 1)
    neg_count = sum(1 for data in train_dataset if data.y.item() == 0)
    other_count = len(train_dataset) - pos_count - neg_count
    print(f"训练集正样本: {pos_count}, 负样本: {neg_count}, 其他标签: {other_count}")
    if pos_count == 0 or neg_count == 0:
        raise ValueError("训练集缺少正样本或负样本，无法训练")

    print("\n开始训练和评估 GAT 模型...")
    model, best_metrics = train_gat(
        train_dataset, val_dataset, test_dataset, num_epochs=args.epoch, batch_size=args.batch_size,
        model_path=args.model_path, device=args.device
    )

    total_elapsed_time = time.time() - total_start_time
    print(f"\n总训练耗时: {total_elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()