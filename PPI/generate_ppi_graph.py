#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从激酶PPI数据（PPI/kinase_ppi_network.csv）构建PyG图对象并保存。
特征：使用Node2Vec从图结构学习128维嵌入。
输出：PPI/kinase_ppi_graph.pt （包含 x, edge_index, node_names, edge_scores 等）
"""

from pathlib import Path

import torch
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from torch_geometric.data import Data

# ==================== 1. 文件路径 ====================
# 以脚本位置定位项目根目录，从任意目录运行脚本时数据路径依然正确
PROJECT_ROOT = Path(__file__).resolve().parents[1]
input_csv = str(PROJECT_ROOT / "PPI/kinase_ppi_network.csv")
output_pt = str(PROJECT_ROOT / "PPI/kinase_ppi_graph.pt")

# ==================== 2. 加载边数据并构建NetworkX图 ====================
print("正在加载边数据...")
df = pd.read_csv(input_csv)
G = nx.Graph()
# 添加边（无向），暂不使用score作为权重
for _, row in df.iterrows():
    a, b = row['preferredName_A'], row['preferredName_B']
    G.add_edge(a, b)

print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

# ==================== 3. 生成节点初始特征（Node2Vec） ====================
print("正在通过Node2Vec生成节点特征（可能需要几分钟）...")
# dimensions=128, walk_length=30, num_walks=200 可根据需要调整
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)
model_n2v = node2vec.fit(window=10, min_count=1, batch_words=4)

# 获取节点列表（保持顺序）
nodes = list(G.nodes())
node_to_idx = {node: i for i, node in enumerate(nodes)}
# 构建特征矩阵 x
x = torch.tensor([model_n2v.wv[node] for node in nodes], dtype=torch.float)
print(f"节点特征维度: {x.shape}")

# ==================== 4. 构建PyG的边索引 ====================
edge_index = []
edge_scores = []  # 保留原始score，可选
for _, row in df.iterrows():
    a, b = row['preferredName_A'], row['preferredName_B']
    score = row['score']
    if a in node_to_idx and b in node_to_idx:
        u, v = node_to_idx[a], node_to_idx[b]
        edge_index.append([u, v])
        edge_index.append([v, u])  # 无向图加反向边
        edge_scores.append(score)
        edge_scores.append(score)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_scores = torch.tensor(edge_scores, dtype=torch.float)
print(f"边索引维度: {edge_index.shape}, 边分数数量: {len(edge_scores)}")

# ==================== 5. 构建节点度数（可选，供PNA使用） ====================
deg = torch.tensor([G.degree(node) for node in nodes], dtype=torch.float)

# ==================== 6. 创建PyG Data对象并保存 ====================
# 保存节点名称映射，方便后续解释结果
data = Data(
    x=x,
    edge_index=edge_index,
    edge_score=edge_scores,      # 边权重（可选）
    deg=deg,                      # 节点度
    num_nodes=len(nodes),
    node_names=nodes,             # 原始基因名列表
    node_to_idx=node_to_idx       # 反向映射（字典）
)

# 由于 PyG 的 Data 对象要求所有属性都是 torch.Tensor，node_names 和 node_to_idx 不是 Tensor，
# 我们可以将其作为普通属性附加，保存时 PyG 会警告但可以保存。
# 更好的做法：保存为单独文件，但为了方便，这里直接保存。
torch.save(data, output_pt)
print(f"图已保存至: {output_pt}")

# 可选的：额外保存节点列表（方便其他工具使用）
node_list_file = str(PROJECT_ROOT / "PPI/kinase_node_list.txt")
with open(node_list_file, 'w') as f:
    for node in nodes:
        f.write(f"{node}\n")
print(f"节点列表已保存至: {node_list_file}")