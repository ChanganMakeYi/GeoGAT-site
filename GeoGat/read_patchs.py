import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import os

def patch_to_data(patch):
    """将补丁数据转换为 PyTorch Geometric 的 Data 对象"""
    try:
        # 提取补丁数据
        node_features = np.array(patch['node_features'], dtype=np.float32)
        edge_index = np.array(patch['subgraph_edges'], dtype=np.int64).T  # 形状 (2, num_edges)
        edge_features = np.array(patch['edge_features'], dtype=np.float32)
        center = patch['center']
        label = int(patch['iface_label'])
        num_nodes = len(patch['subgraph_nodes'])

        # 验证 edge_index
        if edge_index.size > 0:
            max_index = edge_index.max()
            min_index = edge_index.min()
            if max_index >= num_nodes or min_index < 0:
                raise ValueError(
                    f"补丁 patch_{center} 的 edge_index 无效: max={max_index}, min={min_index}, num_nodes={num_nodes}"
                )

        # 标准化节点特征
        node_features = (node_features - node_features.mean(axis=0)) / (node_features.std(axis=0) + 1e-8)

        # 创建 Data 对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
            center=torch.tensor([center], dtype=torch.long)
        )
        return data
    except Exception as e:
        print(f"错误: 处理补丁 patch_{patch['center']} 失败: {e}")
        return None

def read_patches_from_h5(h5_file_path, batch_size=5000, validate=True):
    """分批读取 HDF5 文件中的补丁数据，生成 Data 对象"""
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 文件未找到: {h5_file_path}")

    patches = []
    patch_count = 0
    pos_count = 0
    neg_count = 0

    with h5py.File(h5_file_path, 'r') as f:
        patch_keys = [key for key in f.keys() if key.startswith('patch_')]
        if not patch_keys:
            raise ValueError(f"在 {h5_file_path} 中未找到补丁")

        print(f"发现 {len(patch_keys)} 个补丁")

        for i in tqdm(range(0, len(patch_keys), batch_size), desc="加载补丁"):
            batch_keys = patch_keys[i:i + batch_size]
            batch_patches = []
            for key in batch_keys:
                grp = f[key]
                patch_data = {
                    'center': grp['center'][()],
                    'faces': grp['faces'][:].tolist(),
                    'subgraph_nodes': grp['subgraph_nodes'][:].tolist(),
                    'subgraph_edges': grp['subgraph_edges'][:].tolist(),
                    'node_features': grp['node_features'][:].tolist(),
                    'iface_label': grp['iface_label'][()],
                    'edge_features': grp['edge_features'][:].tolist()
                }
                patch_count += 1
                if patch_data['iface_label'] == 1.0:
                    pos_count += 1
                else:
                    neg_count += 1

                # 转换为 Data 对象
                data = patch_to_data(patch_data)
                if data is not None:
                    batch_patches.append(data)
                else:
                    print(f"跳过无效补丁: {key}")

            print(f"批次 {i//batch_size + 1}: 加载 {len(batch_patches)} 个有效补丁")
            yield batch_patches

        print(f"总计: {patch_count} 个补丁，正样本: {pos_count}，负样本: {neg_count}")

def validate_h5_file(h5_file_path):
    """验证 HDF5 文件中补丁数据的完整性和有效性"""
    with h5py.File(h5_file_path, 'r') as f:
        patch_keys = [key for key in f.keys() if key.startswith('patch_')]
        if not patch_keys:
            print(f"错误: {h5_file_path} 中未找到补丁")
            return False

        for key in tqdm(patch_keys, desc="验证补丁"):
            grp = f[key]
            try:
                center = grp['center'][()]
                num_nodes = len(grp['subgraph_nodes'][:])
                edge_index = grp['subgraph_edges'][:]
                node_features = grp['node_features'][:]
                edge_features = grp['edge_features'][:]
                iface_label = grp['iface_label'][()]

                # 验证 edge_index
                if edge_index.size > 0:
                    max_index = edge_index.max()
                    min_index = edge_index.min()
                    if max_index >= num_nodes or min_index < 0:
                        print(f"错误: {key} 的 edge_index 无效: max={max_index}, min={min_index}, num_nodes={num_nodes}")
                        return False

                # 验证形状
                if node_features.shape[1] != 6:
                    print(f"错误: {key} 的 node_features 形状错误: {node_features}")
                    return False
                if edge_features.shape[1] != 2:
                    print(f"错误: {key} 的 edge_features 形状错误: {edge_features}")
                    return False
                if iface_label not in [0.0, 1.0]:
                    print(f"错误: {key} 的 iface_label 无效: {iface_label}")
                    return False

            except Exception as e:
                print(f"错误: 验证 {key} 失败: {e}")
                return False

        print(f"验证通过: {h5_file_path}，共 {len(patch_keys)} 个补丁")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="读取 HDF5 文件中的补丁数据，转换为 PyTorch Geometric Data 对象",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--h5_file", required=True, help="HDF5 文件路径")
    parser.add_argument("--batch_size", type=int, default=5000, help="分批加载的补丁数量")
    parser.add_argument("--validate_only", action="store_true", help="仅验证 HDF5 文件，不加载补丁")
    args = parser.parse_args()

    # 验证 HDF5 文件
    print(f"验证 HDF5 文件: {args.h5_file}")
    if not validate_h5_file(args.h5_file):
        print("HDF5 文件验证失败，请检查数据")
        return

    if args.validate_only:
        return

    # 加载补丁
    print(f"\n开始加载补丁数据: {args.h5_file}")
    total_patches = 0
    for batch_data in read_patches_from_h5(args.h5_file, batch_size=args.batch_size):
        total_patches += len(batch_data)
        # 示例：打印第一个补丁的详细信息
        if batch_data:
            sample_data = batch_data[0]
            print(f"\n示例补丁 (patch_{sample_data.center.item()}):")
            print(f"  节点特征形状: {sample_data.x}")
            print(f"  边索引形状: {sample_data.edge_index}")
            print(f"  边特征形状: {sample_data.edge_attr}")
            print(f"  标签: {sample_data.y.item()}")
            print(f"  中心面索引: {sample_data.center.item()}")

    print(f"\n总共加载 {total_patches} 个补丁")

if __name__ == "__main__":
    main()