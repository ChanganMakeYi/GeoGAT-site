import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool

class GeoGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, negative_slope=0.2):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.lin = torch.nn.Linear(in_channels, heads * out_channels)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(heads))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, target_index=edge_index[1], num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_attr, target_index, num_nodes):
        # 确保索引合法性
        assert (x_i.size(0) == x_j.size(0)), "x_i 和 x_j 的大小不匹配"
        # 计算注意力系数
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) + self.bias
        # 几何权重：距离和角度
        w_dist = torch.exp(-edge_attr[:, 0] ** 2 / 9.0)  # σ=3Å
        w_angle = edge_attr[:, 1]  # cos(θ)
        alpha = alpha * w_dist.unsqueeze(-1) * w_angle.unsqueeze(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = pyg_softmax(alpha, target_index, num_nodes=num_nodes)
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out.view(-1, self.heads * self.out_channels)

class GeoGATModel(torch.nn.Module):
    def __init__(self, in_channels=6, hidden_channels=128, out_channels=32, heads=4, num_classes=2):
        super().__init__()
        self.conv1 = GeoGATConv(in_channels, hidden_channels, heads)
        self.conv2 = GeoGATConv(hidden_channels * heads, out_channels, heads)
        self.fc = torch.nn.Linear(out_channels * heads, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Add self-loops to handle isolated nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.tensor([[0.0, 1.0]], dtype=torch.float, device=x.device).repeat(x.size(0), 1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0) if edge_attr.numel() > 0 else self_loop_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)  # 全局均值池化
        return self.fc(x)