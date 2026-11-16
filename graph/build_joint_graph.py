#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建联合异构图：同时包含激酶-激酶 PPI 边和激酶-底物磷酸化边。
激酶节点特征使用 Node2Vec（从 PPI 网络学习得到）。
激酶名称只做大写统一，不删除末尾数字。
只保留有 Node2Vec 特征的激酶节点以及有特征的底物节点。
输出：origin_data/joint_hetero_graph.pt
"""

import os
import re
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from torch_geometric.data import HeteroData
import logging

# 以脚本位置定位项目根目录，从任意目录运行脚本时数据路径依然正确
PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 配置路径 ====================
PPI_CSV = str(PROJECT_ROOT / "PPI/kinase_ppi_network.csv")                      # 激酶-激酶 PPI
KINASE_SUBSTRATE_CSV = str(PROJECT_ROOT / "origin_data/kinase_substrate_pairs_detailed.csv")  # 激酶-底物关系
SITE_FEATURE_DIR = str(PROJECT_ROOT / "origin_data/substate")                   # 底物位点特征目录
KINASE_FEATURE_DIRS = [str(PROJECT_ROOT / "origin_data/kinase"), str(PROJECT_ROOT / "origin_data/kinase/out")]  # 激酶特征目录（ESM）
# 全长底物蛋白 ESMC 特征（esmc_extractor.py --fasta Total.fasta 的输出），
# 直接作为 site.x（替换原 31aa 窗口特征）：位点携带蛋白全局上下文
# （docking/无序度/整体家族偏好）；位点局部特异性由 motif 编码器提供
PROTEIN_FEATURE_DIR = str(PROJECT_ROOT / "origin_data/substate/protein_out/")
OUTPUT_GRAPH = str(PROJECT_ROOT / "origin_data/joint_hetero_graph.pt")

# Node2Vec 参数
N2V_DIM = 128
N2V_WALK_LEN = 30
N2V_NUM_WALKS = 200
N2V_WINDOW = 10
N2V_WORKERS = 4

# 底物特征提取时的层配置（与之前一致）
FEATURE_LAYERS = [34, 35, 36]

# ==================== 激酶名称归一化（仅大写，不删数字） ====================
def normalize_kinase_name(raw_name: str) -> str:
    if not isinstance(raw_name, str):
        return ''
    return raw_name.strip().upper()

def load_kinase_features_esm():
    """返回字典 {大写激酶名: ESM特征张量}，与之前相同"""
    feat_dict = {}
    for dir_path in KINASE_FEATURE_DIRS:
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith('.npy'):
                continue
            filepath = os.path.join(dir_path, fname)
            data = np.load(filepath, allow_pickle=True).item()
            orig_name = data.get('name', os.path.splitext(fname)[0])
            norm_name = normalize_kinase_name(orig_name)
            mean_dict = data.get('mean')
            if mean_dict:
                layer_feats = []
                for layer in FEATURE_LAYERS:
                    if layer in mean_dict:
                        layer_feats.append(mean_dict[layer].flatten())
                if layer_feats:
                    feat = np.mean(layer_feats, axis=0)
                    feat_dict[norm_name] = torch.tensor(feat, dtype=torch.float)
            # 如果没有 mean，尝试其他格式（略，与原逻辑一致）
    return feat_dict

# ==================== 加载底物位点特征（同原逻辑） ====================
def load_site_features():
    """返回字典 {site_id: 特征张量}，site_id 格式为 'EPSD_ID_Position'"""
    feat_dict = {}
    for fname in os.listdir(SITE_FEATURE_DIR):
        if not fname.endswith('.npy'):
            continue
        filepath = os.path.join(SITE_FEATURE_DIR, fname)
        data = np.load(filepath, allow_pickle=True).item()
        # 提取特征（与之前一致）
        if isinstance(data, dict):
            mean_dict = data.get('mean')
            if mean_dict:
                layer_feats = []
                for layer in FEATURE_LAYERS:
                    if layer in mean_dict:
                        layer_feats.append(mean_dict[layer].flatten())
                if layer_feats:
                    feat = np.mean(layer_feats, axis=0)
                else:
                    continue
            else:
                # 如果没有 mean，尝试取第一个数值数组
                feat = None
                for v in data.values():
                    if isinstance(v, np.ndarray):
                        feat = v.flatten()
                        break
                if feat is None:
                    continue
        else:
            feat = data.flatten()
        
        # 从文件名解析 EPSD_ID 和 Position
        match = re.search(r'(EP\d+).*?([A-Z]\d+)', fname)
        if match:
            epsd_id = match.group(1)
            position = match.group(2)
            site_id = f"{epsd_id}_{position}"
            feat_dict[site_id] = torch.tensor(feat, dtype=torch.float)
        else:
            logging.warning(f"无法从文件名 {fname} 解析 site_id，跳过")
    logging.info(f"加载了 {len(feat_dict)} 个底物位点的特征")
    return feat_dict

# ==================== 加载全长底物蛋白 ESMC 特征 ====================
def load_substrate_protein_features():
    """返回字典 {EPSD_ID: 1152 维 ESMC mean 特征张量}。

    读取 esmc_extractor.py 对 Total.fasta（全长底物蛋白）的提取输出
    （每个 .npy 含 mean:{34,35,36} 各 1152 维，取三层平均）。
    目录不存在或为空时返回空字典（上游自动退回纯窗口特征）。
    """
    if not os.path.isdir(PROTEIN_FEATURE_DIR):
        raise FileNotFoundError(
            f"全长蛋白特征目录不存在: {PROTEIN_FEATURE_DIR}\n"
            f"先运行: python3 origin_data/esmc_extractor.py "
            f"--fasta origin_data/Total.fasta --out {PROTEIN_FEATURE_DIR}")
    feat_dict = {}
    for fname in os.listdir(PROTEIN_FEATURE_DIR):
        if not fname.endswith('.npy'):
            continue
        data = np.load(os.path.join(PROTEIN_FEATURE_DIR, fname), allow_pickle=True).item()
        mean_dict = data.get('mean') if isinstance(data, dict) else None
        if not mean_dict:
            continue
        layer_feats = [mean_dict[l].flatten() for l in FEATURE_LAYERS if l in mean_dict]
        if not layer_feats:
            continue
        # name 或文件名均为 EPSD_ID（Total.fasta 的 header）
        epsd_id = data.get('name', os.path.splitext(fname)[0])
        feat_dict[epsd_id] = torch.tensor(np.mean(layer_feats, axis=0), dtype=torch.float)
    logging.info(f"加载了 {len(feat_dict)} 个全长底物蛋白的 ESMC 特征")
    return feat_dict

# ==================== 使用 Node2Vec 生成激酶特征 ====================
def generate_kinase_features_from_ppi(ppi_csv, kinase_list):
    """
    从 PPI 数据构建图，利用 Node2Vec 学习激酶节点的嵌入。
    返回 (kinase_feat_dict, all_kinase_nodes)
    - kinase_feat_dict: {激酶名: 特征张量}
    - all_kinase_nodes: 列表，按 node2vec 模型中的顺序（字母序）
    """
    # 1. 读取边并构建 networkx 图
    df_ppi = pd.read_csv(ppi_csv)
    G = nx.Graph()
    df_ppi = pd.read_csv(ppi_csv)
    for _, row in df_ppi.iterrows():
        a = normalize_kinase_name(row['preferredName_A'])
        b = normalize_kinase_name(row['preferredName_B'])
        if a and b:
            G.add_edge(a, b)
    # 只保留 kinase_list 中的节点（但可能某些激酶不在图中）
    nodes_in_graph = [k for k in kinase_list if k in G.nodes]
    if not nodes_in_graph:
        return {}
    # 训练 Node2Vec
    node2vec = Node2Vec(G, dimensions=N2V_DIM, walk_length=N2V_WALK_LEN,
                        num_walks=N2V_NUM_WALKS, workers=N2V_WORKERS)
    model_n2v = node2vec.fit(window=N2V_WINDOW, min_count=1, batch_words=4)
    # 提取特征
    feat_dict = {}
    for node in nodes_in_graph:
        feat_dict[node] = torch.tensor(model_n2v.wv[node], dtype=torch.float)
    return feat_dict

# ==================== 构建联合异构图 ====================
def build_joint_graph():
    # 加载 ESM 特征
    kinase_esm = load_kinase_features_esm()
    logging.info(f"ESM 激酶特征数: {len(kinase_esm)}")
    
    # 加载底物特征（不变）
    site_feat = load_site_features()
    
    # 读取激酶-底物关系，获取所有涉及的激酶（大写归一化后）
    df_ks = pd.read_csv(KINASE_SUBSTRATE_CSV)
    df_ks['Kinase_norm'] = df_ks['Kinase'].astype(str).apply(normalize_kinase_name)
    all_kinases_from_ks = set(df_ks['Kinase_norm'].unique())
    
    # 读取 PPI 边，获取所有激酶（大写归一化后）
    df_ppi = pd.read_csv(PPI_CSV)
    df_ppi['Kinase_A_norm'] = df_ppi['preferredName_A'].astype(str).apply(normalize_kinase_name)
    df_ppi['Kinase_B_norm'] = df_ppi['preferredName_B'].astype(str).apply(normalize_kinase_name)
    all_kinases_from_ppi = set(df_ppi['Kinase_A_norm']) | set(df_ppi['Kinase_B_norm'])
    
    # 候选激酶节点 = 出现在关系或 PPI 中的激酶
    candidate_kinases = all_kinases_from_ks | all_kinases_from_ppi
    
    # 分离：有 ESM 的和没有 ESM 的
    kinases_with_esm = candidate_kinases & set(kinase_esm.keys())
    kinases_without_esm = candidate_kinases - kinases_with_esm
    
    # 为没有 ESM 的激酶生成 Node2Vec 特征
    logging.info(f"需要 Node2Vec 后备的激酶数: {len(kinases_without_esm)}")
    kinase_n2v = generate_kinase_features_from_ppi(PPI_CSV, list(kinases_without_esm))
    logging.info(f"成功生成 Node2Vec 特征的激酶数: {len(kinase_n2v)}")
    
    # 最终特征字典：优先 ESM，后备 Node2Vec
    kinase_feat = {}
    for k in kinases_with_esm:
        kinase_feat[k] = kinase_esm[k]
    for k in kinases_without_esm:
        if k in kinase_n2v:
            kinase_feat[k] = kinase_n2v[k]
    logging.info(f"总激酶节点数（有特征）: {len(kinase_feat)}")
    
    # 过滤关系边：只保留激酶在 kinase_feat 中且底物在 site_feat 中的行
    df_ks['site_id'] = df_ks['EPSD_ID'] + '_' + df_ks['Position'].astype(str)
    mask_ks = df_ks['Kinase_norm'].isin(kinase_feat.keys()) & df_ks['site_id'].isin(site_feat.keys())
    df_ks_filtered = df_ks[mask_ks]
    
    # 过滤 PPI 边：只保留两端激酶都在 kinase_feat 中的边
    mask_ppi = df_ppi['Kinase_A_norm'].isin(kinase_feat.keys()) & df_ppi['Kinase_B_norm'].isin(kinase_feat.keys())
    df_ppi_filtered = df_ppi[mask_ppi]
    
    # 后续构建节点顺序、特征矩阵、边索引等代码保持不变
    # （使用 kinase_feat 中的激酶，按字母排序）
    all_kinases = sorted(kinase_feat.keys())
    kinase_to_idx = {k: i for i, k in enumerate(all_kinases)}
    
    # 获取 ESM 特征维度（如果存在）
    esm_dim = 0
    if kinase_esm:
        sample_key = next(iter(kinase_esm.keys()))
        esm_dim = kinase_esm[sample_key].shape[0]
    
    # 初始化特征矩阵
    esm_x = torch.zeros(len(all_kinases), esm_dim) if esm_dim > 0 else None
    n2v_x = torch.zeros(len(all_kinases), N2V_DIM)
    
    for i, k in enumerate(all_kinases):
        if esm_x is not None and k in kinase_esm:
            esm_x[i] = kinase_esm[k]
        if k in kinase_n2v:
            n2v_x[i] = kinase_n2v[k]
    
    # 底物节点集合来自过滤后的关系（节点集仍由窗口特征目录定义，保证图结构与历史一致）
    all_sites = sorted(df_ks_filtered['site_id'].unique())
    site_to_idx = {s: i for i, s in enumerate(all_sites)}

    # site.x 直接替换为全长蛋白 ESMC 特征（丢弃窗口 mean 特征，位点位点区分交给 motif 编码器）
    protein_feat = load_substrate_protein_features()
    prot_dim = next(iter(protein_feat.values())).shape[0]
    site_prots = [s.rsplit('_', 1)[0] for s in all_sites]
    site_x = torch.zeros(len(all_sites), prot_dim)
    n_missing = 0
    for i, p in enumerate(site_prots):
        if p in protein_feat:
            site_x[i] = protein_feat[p]
        else:
            n_missing += 1
    logging.info(f"site.x = 全长蛋白 ESMC 特征 {prot_dim} 维（已替换窗口特征）| "
                 f"覆盖蛋白 {len(set(site_prots) & set(protein_feat))}/{len(set(site_prots))}，"
                 f"缺失位点 {n_missing}/{len(all_sites)}（零填充）")
    if n_missing > 0.05 * len(all_sites):
        logging.warning("超过 5% 位点的蛋白缺少全长特征——确认 protein_out 提取是否完整")
    
    # 构建 HeteroData 对象
    data = HeteroData()
    # 激酶节点特征：分离存储
    if esm_x is not None:
        data['kinase'].esm_x = esm_x
    data['kinase'].n2v_x = n2v_x
    data['site'].x = site_x
    
    # 边索引构建（与原代码相同）
    edge_ks = []
    for _, row in df_ks_filtered.iterrows():
        u = kinase_to_idx[row['Kinase_norm']]
        v = site_to_idx[row['site_id']]
        edge_ks.append([u, v])
    edge_ks = torch.tensor(edge_ks, dtype=torch.long).t().contiguous()
    
    # 构建 PPI 双向边（注意：df_ppi 需要重新读取或使用之前过滤后的）
    # 此处复用前面已读的 df_ppi 和过滤条件
    # 注意：在前面你已经定义了 mask_ppi 和 df_ppi_filtered，但之后又重新读取了 df_ppi
    # 建议直接使用之前得到的 df_ppi_filtered，避免重复读取
    # 为了清晰，下面使用已经存在的 df_ppi_filtered（需要确保其作用域正确）
    # 如果作用域问题，可以重新读取一次（代价小）
    df_ppi = pd.read_csv(PPI_CSV)
    df_ppi['Kinase_A_norm'] = df_ppi['preferredName_A'].astype(str).apply(normalize_kinase_name)
    df_ppi['Kinase_B_norm'] = df_ppi['preferredName_B'].astype(str).apply(normalize_kinase_name)
    mask_ppi = df_ppi['Kinase_A_norm'].isin(kinase_feat.keys()) & df_ppi['Kinase_B_norm'].isin(kinase_feat.keys())
    df_ppi_filtered = df_ppi[mask_ppi]
    
    edge_ppi = []
    for _, row in df_ppi_filtered.iterrows():
        u = kinase_to_idx[row['Kinase_A_norm']]
        v = kinase_to_idx[row['Kinase_B_norm']]
        if u != v:
            edge_ppi.append([u, v])
            edge_ppi.append([v, u])
    edge_ppi = torch.tensor(edge_ppi, dtype=torch.long).t().contiguous() if edge_ppi else torch.empty((2, 0), dtype=torch.long)
    
    data['kinase', 'phosphorylates', 'site'].edge_index = edge_ks
    data['site', 'rev_phosphorylates', 'kinase'].edge_index = edge_ks[[1, 0], :]
    data['kinase', 'ppis', 'kinase'].edge_index = edge_ppi
    
    # 保存节点名称
    data['kinase'].node_names = all_kinases
    data['site'].node_names = all_sites
    
    # 9. 保存
    torch.save(data, OUTPUT_GRAPH)
    logging.info(f"联合异构图已保存至 {OUTPUT_GRAPH}")
    logging.info(f"图结构:\n{data}")
    
    return data

if __name__ == "__main__":
    build_joint_graph()