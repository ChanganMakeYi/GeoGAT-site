from torchviz import make_dot
from torch_geometric.data import Data
import torch
from gat import GeoGATModel
# 创建示例数据
x = torch.randn(10, 6)  # 10 个节点，6 维特征
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # 示例边
edge_attr = torch.randn(4, 2)  # 4 条边，2 维属性
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# 初始化模型
model = GeoGATModel(in_channels=6, hidden_channels=128, out_channels=32, heads=4, num_classes=2)

# 前向传播
model.eval()
out = model(data)

# 生成结构图
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render("GeoGATModel", format="png")  # 保存为 PNG