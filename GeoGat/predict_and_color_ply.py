import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import h5py
import os
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from gat import GeoGATModel
from dataprocess_ply import parse_ply_file, load_patches_from_h5


def interpolate_color(score):
    """根据预测分数（0到1）插值颜色，从蓝色->白色->红色"""
    score = np.clip(score, 0.0, 1.0)
    if score <= 0.5:
        t = score / 0.5
        r = int(255 * t)
        g = int(255 * t)
        b = int(255 * (1 - t) + 255 * t)
    else:
        t = (score - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - t))
        b = int(255 * (1 - t))
    return r, g, b


def predict_patches(model, h5_file_path, batch_size=32, device='cpu'):
    """对补丁进行预测，返回中心面索引、预测概率和真实标签"""
    model.eval()
    predictions = []
    center_indices = []
    iface_labels = []

    # 检查 HDF5 文件结构
    with h5py.File(h5_file_path, 'r') as f:
        print("HDF5 文件键:", list(f.keys())[:10], "...")
        patch_keys = [key for key in f.keys() if key.startswith('patch_')]
        if not patch_keys:
            raise KeyError("HDF5 文件中未找到补丁组（patch_<i>），请检查生成脚本")

        num_patches = len(patch_keys)
        print(f"HDF5 补丁数量: {num_patches}")

        try:
            iface_labels = [int(f[key]['iface_label'][()]) for key in patch_keys]
            print(f"HDF5 标签分布: 正例={sum(iface_labels)}, 负例={len(iface_labels) - sum(iface_labels)}")
            print(f"HDF5 前10个标签: {iface_labels[:10]}")
        except Exception as e:
            print(f"错误: 无法读取 patch_<i>/iface_label 数据: {e}")
            raise

        print(f"patch_0/node_features 形状: {f['patch_0']['node_features'].shape}")
        print(f"patch_0/edge_features 形状: {f['patch_0']['edge_features'].shape}")

    batch_idx = 0
    for batch_data in load_patches_from_h5(h5_file_path, batch_size=batch_size):
        print(f"处理批次 {batch_idx + 1}, 包含 {len(batch_data)} 个补丁")
        if not batch_data:
            print("警告: 批次为空，跳过")
            continue
        print(f"第一个补丁: x={batch_data[0].x.shape}, edge_index={batch_data[0].edge_index.shape}, "
              f"edge_attr={batch_data[0].edge_attr.shape}, y={batch_data[0].y.shape}, center={getattr(batch_data[0], 'center', 'None')}")
        loader = DataLoader(batch_data, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                try:
                    batch_center_indices = [batch_data[i].center.item() for i in range(len(batch_data))]
                    batch_labels = [batch_data[i].y.item() for i in range(len(batch_data))]
                    print(f"批次 {batch_idx + 1} 中心面索引: {batch_center_indices[:5]}...")
                    print(f"批次 {batch_idx + 1} 加载标签: {batch_labels[:5]}...")
                except AttributeError as e:
                    print(f"警告: batch_data 缺少属性 {e}, 使用补丁索引作为中心面索引")
                    batch_center_indices = list(range(batch_idx * batch_size, batch_idx * batch_size + len(batch_data)))
                    batch_labels = [0] * len(batch_data)
                predictions.extend(probs)
                center_indices.extend(batch_center_indices)
        batch_idx += 1

    print(f"预测完成: {len(predictions)} 个补丁")
    if len(predictions) != len(iface_labels):
        print(f"错误: 预测数量 ({len(predictions)}) 与标签数量 ({len(iface_labels)}) 不匹配")
        raise ValueError("预测和标签数量不一致")
    print(f"预测概率 (前10): {predictions[:10]}")
    return center_indices, predictions, iface_labels


def map_scores_to_vertices(faces, center_indices, predictions, num_vertices):
    """将补丁中心面的预测分数映射到顶点"""
    vertex_scores = np.zeros(num_vertices)
    vertex_counts = np.zeros(num_vertices)

    for face_idx, score in zip(center_indices, predictions):
        if face_idx >= len(faces):
            print(f"警告: 中心面索引 {face_idx} 超出面数组范围")
            continue
        for vertex_idx in faces[face_idx]:
            if vertex_idx >= num_vertices:
                print(f"警告: 顶点索引 {vertex_idx} 超出顶点数组范围")
                continue
            vertex_scores[vertex_idx] += score
            vertex_counts[vertex_idx] += 1

    vertex_counts[vertex_counts == 0] = 1
    vertex_scores = vertex_scores / vertex_counts
    return vertex_scores


def update_ply_with_colors(ply_file_path, output_ply_file, vertex_scores, center_indices, predictions):
    """更新 PLY 文件，添加顶点颜色和面预测分数"""
    with open(ply_file_path, 'r', encoding='ascii') as f:
        lines = f.readlines()

    header = []
    vertex_count = 0
    face_count = 0
    in_header = True
    for i, line in enumerate(lines):
        if in_header:
            header.append(line)
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            if line.startswith('element face'):
                face_count = int(line.split()[2])
            if line == 'end_header\n':
                header_end_idx = i + 1
                in_header = False

    vertex_lines = lines[header_end_idx:header_end_idx + vertex_count]
    face_lines = lines[header_end_idx + vertex_count:header_end_idx + vertex_count + face_count]

    if len(vertex_lines) != vertex_count:
        print(f"实际顶点行: {len(vertex_lines)}")
        print("前 5 行顶点数据:", vertex_lines[:5])
        print("前 5 行后续数据:", lines[header_end_idx + vertex_count:header_end_idx + vertex_count + 5])
        raise ValueError(f"顶点数量不匹配: 头部声明 {vertex_count}, 实际 {len(vertex_lines)}")

    if len(face_lines) != face_count:
        print(f"实际面行: {len(face_lines)}")
        print("前 5 行面数据:", face_lines[:5])
        raise ValueError(f"面数量不匹配: 头部声明 {face_count}, 实际 {len(face_lines)}")

    new_header = []
    for line in header:
        if line.startswith('property float iface') or line.startswith('property float red') or line.startswith(
                'property uchar red'):
            continue
        new_header.append(line)
        if line.startswith('element vertex'):
            new_header.append('property uchar red\n')
            new_header.append('property uchar green\n')
            new_header.append('property uchar blue\n')
        if line.startswith('element face'):
            new_header.append('property float predict_score\n')

    new_vertex_lines = []
    for i, line in enumerate(vertex_lines):
        parts = line.strip().split()
        if len(parts) < 9:
            print(f"警告: 顶点 {i} 数据不完整，跳过: {line.strip()}")
            continue
        vertex_data = parts[:9]
        r, g, b = interpolate_color(vertex_scores[i])
        vertex_data.extend([str(int(r)), str(int(g)), str(int(b))])
        new_vertex_lines.append(' '.join(vertex_data) + '\n')

    new_face_lines = []
    face_scores = [0.0] * face_count
    for center_idx, pred in zip(center_indices, predictions):
        if center_idx < face_count:
            face_scores[center_idx] = pred
        else:
            print(f"警告: 中心面索引 {center_idx} 超出面数组范围")
    print(f"非零面预测分数数量: {sum(1 for s in face_scores if s > 0)}")
    print(f"前 5 个面预测分数: {face_scores[:5]}")
    for i, line in enumerate(face_lines):
        parts = line.strip().split()
        if len(parts) < 4 or parts[0] != '3':
            print(f"警告: 面数据不完整或非三角面: {line.strip()}")
            continue
        face_data = parts
        face_data.append(f"{face_scores[i]:.6f}")
        new_face_lines.append(' '.join(face_data) + '\n')

    with open(output_ply_file, 'w', encoding='ascii') as f:
        f.writelines(new_header)
        f.writelines(new_vertex_lines)
        f.writelines(new_face_lines)

    print(f"已保存带颜色标注和面预测分数的 PLY 文件: {output_ply_file}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 GeoGAT 模型预测 PLY 文件的界面概率，并按分数标注颜色",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ply_file", required=True, help="输入 PLY 文件路径")
    parser.add_argument("--h5_file", required=True, help="补丁数据的 HDF5 文件路径")
    parser.add_argument("--model_path", default="geogat_model.pth", help="训练好的模型路径")
    parser.add_argument("--output_ply", default="output_colored.ply", help="输出带颜色标注的 PLY 文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="预测时的批大小")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu 或 cuda)")
    args = parser.parse_args()

    start_time = time.time()

    try:
        vertices, faces = parse_ply_file(args.ply_file)
        print(f"解析完成: {len(vertices)} 个顶点, {len(faces)} 个面")
    except Exception as e:
        print(f"错误: 解析 PLY 文件失败: {e}")
        return

    model = GeoGATModel(in_channels=6, hidden_channels=128, out_channels=32, heads=4, num_classes=2)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.to(args.device)
        print(f"模型加载成功: {args.model_path}")
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        return

    try:
        center_indices, predictions, iface_labels = predict_patches(model, args.h5_file, batch_size=args.batch_size,
                                                                    device=args.device)
    except Exception as e:
        print(f"错误: 预测补丁失败: {e}")
        return

    try:
        if len(np.unique(iface_labels)) > 1:
            auc_score = roc_auc_score(iface_labels, predictions)
            aupr_score = average_precision_score(iface_labels, predictions)
            print(f"AUC: {auc_score:.4f}")
            print(f"AUPR: {aupr_score:.4f}")
        else:
            print("警告: 真实标签仅包含单一类别，无法计算 AUC 和 AUPR")
            print(f"标签分布: 正例={sum(iface_labels)}, 负例={len(iface_labels) - sum(iface_labels)}")
            print(f"真实标签 (前10): {iface_labels[:10]}")
            print(f"预测概率 (前10): {predictions[:10]}")
    except ValueError as e:
        print(f"错误: 计算 AUC/AUPR 失败: {e}")
        print(f"真实标签 (前10): {iface_labels[:10]}")
        print(f"预测概率 (前10): {predictions[:10]}")

    vertex_scores = map_scores_to_vertices(faces, center_indices, predictions, len(vertices))
    print(f"顶点分数统计: 平均={vertex_scores.mean():.4f}, 最小={vertex_scores.min():.4f}, 最大={vertex_scores.max():.4f}")

    try:
        update_ply_with_colors(args.ply_file, args.output_ply, vertex_scores, center_indices, predictions)
    except Exception as e:
        print(f"错误: 保存 PLY 文件失败: {e}")
        return

    elapsed_time = time.time() - start_time
    print(f"总处理时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()