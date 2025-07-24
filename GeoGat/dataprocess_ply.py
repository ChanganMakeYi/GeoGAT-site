import numpy as np
import networkx as nx
from scipy.spatial import distance
import random
from multiprocessing import Pool, Array
import time
import os
import h5py
from collections import defaultdict
from torch_geometric.data import Data
import torch
import glob

def parse_ply_file(ply_file_path):
    """解析 PLY 文件，验证面索引，适配格式 [x, y, z, charge, hbond, hphob, iface, nx, ny, nz]"""
    if not os.path.exists(ply_file_path):
        raise FileNotFoundError(f"PLY 文件未找到: {ply_file_path}")

    vertices = []
    faces = []
    in_header = True

    with open(ply_file_path, 'r', encoding='ascii') as f:
        lines = f.readlines()
        if not lines:
            raise ValueError("PLY 文件为空")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == 'end_header':
                in_header = False
                continue
            if in_header:
                continue
            parts = line.split()
            if len(parts) == 10:  # 新格式有 10 个属性
                try:
                    vertex_data = [float(x) for x in parts]
                    vertices.append(vertex_data)  # [x, y, z, charge, hbond, hphob, iface, nx, ny, nz]
                except ValueError:
                    print(f"警告: 跳过无效顶点行: {line}")
            elif len(parts) == 4 and parts[0] == '3':
                try:
                    _, i, j, k = map(int, parts)
                    faces.append([i, j, k])
                except ValueError:
                    print(f"警告: 跳过无效面行: {line}")

    if not vertices:
        raise ValueError("未从 PLY 文件中解析到顶点")
    vertices = np.array(vertices)
    faces = np.array(faces)
    if len(faces) == 0:
        print(f"警告: 未从 PLY 文件中解析到面: {ply_file_path}")
    else:
        max_face_index = faces.max()
        if max_face_index >= len(vertices):
            raise ValueError(f"面索引超出顶点范围: 最大索引 {max_face_index}, 顶点数 {len(vertices)}")
        for i, face in enumerate(faces):
            if any(idx < 0 for idx in face):
                raise ValueError(f"面 {i} 包含负索引: {face}")
            if len(set(face)) != 3:
                print(f"警告: 面 {i} 包含重复或无效顶点索引: {face}")

    print(f"解析完成: {ply_file_path}, {len(vertices)} 个顶点, {len(faces)} 个面")
    return vertices, faces

def create_face_adjacency(faces):
    """高效构建面邻接关系，使用顶点到面映射，优化内存"""
    vertex_to_faces = defaultdict(list)
    for i, face in enumerate(faces):
        for v in face:
            vertex_to_faces[v].append(i)

    face_to_neighbors = defaultdict(set)
    for v in vertex_to_faces:
        faces_sharing_vertex = vertex_to_faces[v]
        for i in faces_sharing_vertex:
            for j in faces_sharing_vertex:
                if i != j and len(set(faces[i]) & set(faces[j])) == 2:
                    face_to_neighbors[i].add(j)

    return {i: list(neighbors) for i, neighbors in face_to_neighbors.items()}

def compute_face_centers(vertices, faces):
    """预计算所有面中心坐标"""
    face_centers = np.array([np.mean([vertices[v][:3] for v in face], axis=0) for face in faces])
    return face_centers

def compute_face_patch(vertices, faces, face_centers, face_adjacency, center_face_idx, max_radius=9.0):
    try:
        center_pos = face_centers[center_face_idx]
        distances = np.linalg.norm(face_centers - center_pos, axis=1)
        patch_face_indices = np.where(distances <= max_radius)[0].tolist()

        if not patch_face_indices:
            print(f"警告: 中心面 {center_face_idx} 的补丁为空")
            return patch_face_indices, [], {}, 0.0, []

        patch_subgraph = []
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(patch_face_indices)}
        node_features = []
        edge_features = []
        for i in patch_face_indices:
            face = faces[i]
            face_vertices = [vertices[v] for v in face]
            face_features = np.mean([np.concatenate((v[7:10], v[3:6])) for v in face_vertices], axis=0)
            node_features.append(face_features)

        for i in patch_face_indices:
            for j in face_adjacency.get(i, []):
                if j in patch_face_indices and i < j:
                    if i not in node_mapping or j not in node_mapping:
                        print(f"错误: 面 {i} 或 {j} 不在 node_mapping 中，patch_face_indices={patch_face_indices}")
                        continue
                    edge_dist = distance.euclidean(face_centers[i], face_centers[j])
                    ni = np.mean([vertices[v][7:10] for v in faces[i]], axis=0)
                    nj = np.mean([vertices[v][7:10] for v in faces[j]], axis=0)
                    cos_angle = np.dot(ni, nj) / (np.linalg.norm(ni) * np.linalg.norm(nj) + 1e-8)
                    patch_subgraph.append((node_mapping[i], node_mapping[j]))
                    edge_features.append([edge_dist, cos_angle])

        center_face_vertices = [vertices[v] for v in faces[center_face_idx]]
        iface_label = 1.0 if any(v[6] >= 1.0 for v in center_face_vertices) else 0.0

        # 验证 patch_subgraph 的索引
        if patch_subgraph:
            max_index = max(max(edge[0], edge[1]) for edge in patch_subgraph)
            if max_index >= len(patch_face_indices):
                print(f"错误: patch_subgraph 包含无效索引，最大索引={max_index}，节点数={len(patch_face_indices)}")
                return [], [], [], 0.0, []

        return patch_face_indices, patch_subgraph, node_features, iface_label, edge_features
    except Exception as e:
        print(f"错误: 处理中心面 {center_face_idx} 失败: {e}")
        return [], [], [], 0.0, []

def compute_face_patch_for_face(args):
    """并行处理单个面的补丁计算"""
    vertices, faces, face_centers, face_adjacency, center_face_idx, max_radius = args
    try:
        patch_face_indices, patch_subgraph, node_features, iface_label, edge_features = compute_face_patch(
            vertices, faces, face_centers, face_adjacency, center_face_idx, max_radius
        )
        return center_face_idx, patch_face_indices, patch_subgraph, node_features, iface_label, edge_features
    except Exception as e:
        print(f"错误: 并行处理中心面 {center_face_idx} 失败: {e}")
        return center_face_idx, [], [], [], 0.0, []

def generate_sampled_face_patches_parallel(vertices, faces, sample_size=None, max_radius=9.0, num_workers=4,
                                          output_dir="face_patches", batch_size=100):
    """分批并行生成补丁，逐批写入 HDF5，优化内存"""
    if not faces.size:
        raise ValueError("面数组为空，无法生成补丁")

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            raise RuntimeError(f"无法创建输出目录 {output_dir}: {e}")

    available_faces = len(faces)
    if sample_size is None:
        sampled_faces = list(range(available_faces))
        sample_size = available_faces
        print(f"使用全部 {sample_size} 个面生成补丁")
    else:
        sample_size = min(sample_size, available_faces)
        sampled_faces = random.sample(list(range(available_faces)), sample_size)
        print(f"从 {available_faces} 个面中采样 {sample_size} 个生成补丁")

    face_centers = compute_face_centers(vertices, faces)
    face_adjacency = create_face_adjacency(faces)

    vertices_shared = Array('d', vertices.ravel())
    vertices_np = np.frombuffer(vertices_shared.get_obj(), dtype=np.float64).reshape(vertices.shape)
    faces_shared = Array('i', faces.ravel())
    faces_np = np.frombuffer(faces_shared.get_obj(), dtype=np.int32).reshape(faces.shape)
    face_centers_shared = Array('d', face_centers.ravel())
    face_centers_np = np.frombuffer(face_centers_shared.get_obj(), dtype=np.float64).reshape(face_centers.shape)

    output_h5 = os.path.join(output_dir, "patches.h5")
    start_time = time.time()
    patch_count = 0

    with h5py.File(output_h5, 'w') as f:
        for i in range(0, len(sampled_faces), batch_size):
            batch_faces = sampled_faces[i:i + batch_size]
            tasks = [(vertices_np, faces_np, face_centers_np, face_adjacency, idx, max_radius) for idx in batch_faces]

            with Pool(num_workers) as pool:
                results = pool.imap(compute_face_patch_for_face, tasks, chunksize=max(1, len(tasks) // num_workers))
                for center_face_idx, patch_face_indices, patch_subgraph, node_features, iface_label, edge_features in results:
                    if not patch_face_indices:
                        print(f"跳过空补丁: 中心面 {center_face_idx}")
                        continue
                    patch_data = {
                        'center': center_face_idx,
                        'faces': patch_face_indices,
                        'subgraph_nodes': patch_face_indices,
                        'subgraph_edges': patch_subgraph,  # 已使用本地索引
                        'node_features': node_features,
                        'iface_label': iface_label,
                        'edge_features': edge_features
                    }
                    grp = f.create_group(f'patch_{center_face_idx}')
                    grp.create_dataset('center', data=patch_data['center'])
                    grp.create_dataset('faces', data=patch_data['faces'])
                    grp.create_dataset('subgraph_nodes', data=patch_data['subgraph_nodes'])
                    grp.create_dataset('subgraph_edges', data=patch_data['subgraph_edges'])
                    grp.create_dataset('node_features', data=np.array(patch_data['node_features'], dtype=np.float32))
                    grp.create_dataset('iface_label', data=patch_data['iface_label'])
                    grp.create_dataset('edge_features', data=np.array(patch_data['edge_features'], dtype=np.float32))
                    patch_count += 1
                    print(f"保存补丁: patch_{center_face_idx}，包含 {len(patch_face_indices)} 个面, 标签: {iface_label}")

            del tasks
            import gc
            gc.collect()

    elapsed_time = time.time() - start_time
    print(f"在 {elapsed_time:.2f} 秒内生成 {patch_count} 个补丁，保存到 {output_h5}")
    return patch_count

def patch_to_data(patch):
    """将补丁转换为 PyTorch Geometric 的 Data 对象，使用 iface_label 作为标签"""
    node_features = np.array(patch['node_features'], dtype=np.float32)
    if len(patch['subgraph_edges']) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty((0, 2), dtype=np.float32)
    else:
        edge_index = np.array(patch['subgraph_edges'], dtype=np.int64).T
        edge_features = np.array(patch['edge_features'], dtype=np.float32)

    num_nodes = len(patch['subgraph_nodes'])
    if num_nodes == 0:
        print(f"错误: 补丁 {patch['center']} 无节点，跳过")
        return None

    # 验证 edge_index
    if edge_index.size > 0:
        max_index = edge_index.max()
        min_index = edge_index.min()
        if max_index >= num_nodes or min_index < 0:
            print(f"错误: 补丁 {patch['center']} 的 edge_index 无效: max={max_index}, min={min_index}, num_nodes={num_nodes}")
            # 尝试修复：只保留有效边
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & (edge_index[0] >= 0) & (edge_index[1] >= 0)
            if valid_mask.sum() == 0:
                print(f"警告: 补丁 {patch['center']} 无有效边，将作为无边图处理")
                edge_index = np.empty((2, 0), dtype=np.int64)
                edge_features = np.empty((0, 2), dtype=np.float32)
            else:
                edge_index = edge_index[:, valid_mask]
                edge_features = edge_features[valid_mask]
                print(f"修复: 保留 {valid_mask.sum()} 条有效边，原边数 {len(patch['subgraph_edges'])}")

    # 修复后再次验证
    if edge_index.size > 0:
        max_index = edge_index.max()
        min_index = edge_index.min()
        if max_index >= num_nodes or min_index < 0:
            print(f"错误: 补丁 {patch['center']} 修复后仍无效 edge_index: max={max_index}, min={min_index}, num_nodes={num_nodes}")
            return None

    # 检查边数量是否过多
    num_edges = edge_index.shape[1]
    if num_edges > 10000:  # 任意阈值，根据需要调整
        print(f"错误: 补丁 {patch['center']} 有过多边 {num_edges}，可能网格异常，跳过")
        return None

    # 标准化节点特征
    if num_nodes > 1:
        mean = node_features.mean(axis=0)
        std = node_features.std(axis=0) + 1e-8
        node_features = (node_features - mean) / std
    else:
        node_features = node_features - node_features.mean(axis=0)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    label = int(patch['iface_label'])

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), edge_attr=edge_attr)
    data.center = torch.tensor([patch['center']], dtype=torch.long)  # 保存中心面索引
    return data

# ... (rest of the code remains the same)

def load_patches_from_h5(h5_file_path, batch_size=5000):
    """分批从 HDF5 文件加载补丁数据，优化内存"""
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 文件未找到: {h5_file_path}")

    patches = []
    with h5py.File(h5_file_path, 'r') as f:
        patch_keys = [key for key in f.keys() if key.startswith('patch_')]
        if not patch_keys:
            raise ValueError(f"在 {h5_file_path} 中未找到补丁")

        for i in range(0, len(patch_keys), batch_size):
            batch_keys = patch_keys[i:i + batch_size]
            for key in batch_keys:
                grp = f[key]
                required_datasets = ['center', 'faces', 'subgraph_nodes', 'subgraph_edges', 'node_features',
                                     'iface_label', 'edge_features']
                missing = [ds for ds in required_datasets if ds not in grp]
                if missing:
                    print(f"警告: 补丁 {key} 缺少数据集 {missing}，跳过")
                    continue
                patch_data = {
                    'center': grp['center'][()],
                    'faces': grp['faces'][:].tolist(),
                    'subgraph_nodes': grp['subgraph_nodes'][:].tolist(),
                    'subgraph_edges': grp['subgraph_edges'][:].tolist(),
                    'node_features': grp['node_features'][:].tolist(),
                    'iface_label': grp['iface_label'][()],
                    'edge_features': grp['edge_features'][:].tolist()
                }
                patches.append(patch_data)
            print(f"加载 {len(patches)} 个补丁")
            batch_converted = [patch_to_data(patch) for patch in patches]
            yield [d for d in batch_converted if d is not None]
            patches = []

        if patches:
            batch_converted = [patch_to_data(patch) for patch in patches]
            yield [d for d in batch_converted if d is not None]

def process_ply_file(ply_file_path, sample_size, num_workers, output_dir, label, max_radius, batch_size):
    """处理单个 PLY 文件，生成以面为中心的补丁"""
    start_time = time.time()
    print(f"\n处理 {label} PLY 文件: {ply_file_path}")

    try:
        vertices, faces = parse_ply_file(ply_file_path)
    except Exception as e:
        print(f"错误: 解析 {label} PLY 文件失败: {e}")
        return None, 0
    print(f"加载 {len(vertices)} 个顶点和 {len(faces)} 个面，耗时 {time.time() - start_time:.2f} 秒")

    print(f"生成 {label} 补丁...")
    try:
        patch_count = generate_sampled_face_patches_parallel(
            vertices, faces, sample_size=sample_size, max_radius=max_radius, num_workers=num_workers,
            output_dir=output_dir, batch_size=batch_size
        )
    except Exception as e:
        print(f"错误: 生成 {label} 补丁失败: {e}")
        return None, 0

    elapsed_time = time.time() - start_time
    print(f"{label} 总处理时间: {elapsed_time:.2f} 秒")
    return None, patch_count

def generate_training_dataset(ply_files, sample_size, num_workers, output_base_dir, label, max_radius, batch_size):
    """批量处理多个 PLY 文件，生成训练数据集并保存到 HDF5 文件"""
    total_patch_count = 0
    for ply_file in ply_files:
        ply_dir = os.path.dirname(ply_file)
        base_name = os.path.splitext(os.path.basename(ply_file))[0]
        if base_name.endswith("_updated"):
            base_name = base_name[:-8]
        output_dir = os.path.join(output_base_dir, f"{base_name}_face_patches_complex")

        _, patch_count = process_ply_file(
            ply_file_path=ply_file,
            sample_size=sample_size,
            num_workers=num_workers,
            output_dir=output_dir,
            label=f"{label}_{base_name}",
            max_radius=max_radius,
            batch_size=batch_size
        )
        total_patch_count += patch_count
        print(f"处理 {ply_file} 完成，生成补丁数量: {patch_count}")

    print(f"总共生成补丁数量: {total_patch_count}")
    return total_patch_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="从多个 PLY 文件生成训练数据，优化内存使用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ply_dir", required=True, help="PLY 文件所在目录")
    parser.add_argument("--sample_size", type=int, default=None, nargs='?', help="采样面数量，None 表示完全采样")
    parser.add_argument("--num_workers", type=int, default=4, help="并行处理进程数")
    parser.add_argument("--max_radius", type=float, default=9.0, help="补丁最大半径（Å）")
    parser.add_argument("--output_base_dir", default="face_patches", help="补丁输出基础目录")
    parser.add_argument("--label", default="training", help="处理标签")
    parser.add_argument("--batch_size", type=int, default=3000, help="每批处理的补丁数量")
    args = parser.parse_args()

    ply_files = glob.glob(os.path.join(args.ply_dir, "*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"在目录 {args.ply_dir} 中未找到 PLY 文件")

    print(f"找到 {len(ply_files)} 个 PLY 文件: {ply_files}")
    patch_count = generate_training_dataset(
        ply_files=ply_files,
        sample_size=args.sample_size,
        num_workers=args.num_workers,
        output_base_dir=args.output_base_dir,
        label=args.label,
        max_radius=args.max_radius,
        batch_size=args.batch_size
    )
    print(f"总生成补丁数量: {patch_count}")