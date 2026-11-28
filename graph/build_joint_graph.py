#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建联合异构图：同时包含激酶-激酶 PPI 边和激酶-底物磷酸化边。
激酶节点名 = HGNC 现用基因符号：KSR / PPI / ESM 特征里的原始名经别名表归一
（data_prep/build_kinase_alias_map.py 生成），同一激酶的旧名与现用名
（CDC2 与 CDK1、FRAP 与 MTOR…）合并为一个节点。
KSR 里无法归属到某个具体激酶的注释也尽量保留，用边层级 edge_level 区分（见 data_prep/edge_levels.py）：
  0  特定激酶的注释
  1  家族层面注释（PKA_GROUP、CAMKII…）：挂到 FAMILY_<家族> 节点上，特征取成员特征的质心，只用于训练
  2  自磷酸化（AUTOCATALYSIS）：按底物蛋白反推出激酶，只用于训练
仍然不进图的只有：非蛋白激酶、融合蛋白 / 注释文字、没有成员定义的泛称、反推不出激酶的自磷酸化。
只保留有 ESM 特征或 Node2Vec 特征的激酶节点以及有特征的底物节点。
输出：origin_data/joint_hetero_graph.pt
"""

import os
import re
import sys
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.kinase_names import ALIAS_MAP_PATH, EXCLUDED_ENTITY_TYPES, FAMILY_PREFIX, KinaseNamer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 配置路径 ====================
PPI_CSV = str(PROJECT_ROOT / "PPI/kinase_ppi_network.csv")                      # 激酶-激酶 PPI
KINASE_SUBSTRATE_CSV = str(PROJECT_ROOT / "origin_data/kinase_substrate_pairs_detailed.csv")  # 激酶-底物关系
# 自磷酸化位点 -> 激酶（build_kinase_alias_map.py 按底物的 UniProt 反推生成）
AUTOPHOS_MAP_CSV = str(PROJECT_ROOT / "origin_data/kinase_autophos_map.csv")
SITE_FEATURE_DIR = str(PROJECT_ROOT / "origin_data/substate")                   # 底物位点特征目录
KINASE_FEATURE_DIRS = [str(PROJECT_ROOT / "origin_data/kinase"), str(PROJECT_ROOT / "origin_data/kinase/out")]  # 激酶特征目录（ESM）
# 激酶序列 FASTA（三者名字不重叠），用于判定特征文件名字的命名语义，以及同一规范名有多个
# 特征文件时的取舍顺序：UniProt 按基因名检索（现用符号语义、序列较新）在前，Manning 2002 在后
KINASE_FASTAS = [str(PROJECT_ROOT / "origin_data/kinase_sequences_uniprot.fasta"),
                 str(PROJECT_ROOT / "origin_data/kinase_sequences.fasta"),
                 str(PROJECT_ROOT / "origin_data/kinase_sequences_out.fasta")]
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

# ==================== 激酶名称归一（原始名 -> HGNC 现用符号） ====================
NAMER = KinaseNamer.load(ALIAS_MAP_PATH)   # 缺别名表时在此报错并提示先生成

def normalize_kinase_name(raw_name: str) -> str:
    """原始名的字面归一（去空白、转大写），不含别名合并。"""
    if not isinstance(raw_name, str):
        return ''
    return raw_name.strip().upper()

def canonical_kinase_name(raw_name: str, semantics: str = 'kinbase') -> str:
    """原始名 -> 规范名（HGNC 现用符号）；用于 PPI 表与 ESM 特征文件的名字（KSR 表走 classify_kinase_rows）。
    非蛋白激酶、融合蛋白 / 注释文字等不进图的类型返回空串。
    semantics: KSR 表与 KinBase 命名的序列名用 'kinbase'；STRING 的 preferredName 与 UniProt 按基因名
    检索得到的序列名用 'symbol'——同一个字符串在两种命名里可能是不同的蛋白（GCK、MST1、PDK1…）。"""
    name, etype = NAMER.info(raw_name, semantics=semantics)
    return '' if etype in EXCLUDED_ENTITY_TYPES else name

def _fasta_header_names(path):
    names = set()
    if os.path.exists(path):
        for line in open(path):
            if line.startswith('>'):
                names.add(normalize_kinase_name(line[1:].split()[0]))
    return names

_FASTA_NAMES = [_fasta_header_names(p) for p in KINASE_FASTAS]

def _name_semantics(raw_name: str) -> str:
    """UniProt 按基因名检索得到的序列名（KINASE_FASTAS[0]）是现用符号语义，其余按 KinBase 语义。"""
    return 'symbol' if normalize_kinase_name(raw_name) in _FASTA_NAMES[0] else 'kinbase'

def _feature_rank(raw_name: str, canonical: str) -> tuple:
    """同一规范名有多个特征文件（旧名与现用名的序列版本略有差异）时的取舍：
    规范名本身的特征优先，其余按序列来源的新旧（见 KINASE_FASTAS 的顺序），再按名字字典序，
    保证结果与目录遍历顺序无关。"""
    name = normalize_kinase_name(raw_name)
    if name == canonical:
        return (0, 0, '')
    for rank, names in enumerate(_FASTA_NAMES, start=1):
        if name in names:
            return (1, rank, name)
    return (1, len(_FASTA_NAMES) + 1, name)


def load_autophos_map(path: str) -> dict:
    """{位点 id: 激酶规范名}。文件不存在时返回空字典（此时自磷酸化的行全部不进图）。"""
    if not os.path.exists(path):
        logging.warning(f"找不到自磷酸化对照表 {path}，AUTOCATALYSIS 的行不会进图；"
                        f"先运行 python3 data_prep/build_kinase_alias_map.py 生成")
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return dict(zip(df['site_id'], df['kinase']))


# 边层级（与 data_prep/edge_levels.py 一致）
LEVEL_SPECIFIC, LEVEL_FAMILY, LEVEL_AUTOPHOS = 0, 1, 2

def classify_kinase_rows(df_ks, autophos: dict):
    """给 KSR 的每一行确定图中的激酶节点与边层级。返回 (节点列表, 层级列表)：
        LEVEL_SPECIFIC  特定激酶的注释（未解析的旧名按原名成节点，与以前一致）
        LEVEL_FAMILY    家族层面注释，节点是 FAMILY_xxx（特征取成员质心，只用于训练）
        LEVEL_AUTOPHOS  自磷酸化，激酶由底物位点查 autophos 反推（只用于训练）
    节点为空串表示不进图：没有家族定义的泛称、非蛋白激酶、融合蛋白 / 注释文字、反推不出激酶的自磷酸化。"""
    nodes, levels = [], []
    for name, site in zip(df_ks['Kinase'], df_ks['site_id']):
        node, etype = NAMER.info(name)
        if etype == 'autocatalysis':
            node, level = autophos.get(site, ''), LEVEL_AUTOPHOS
        elif etype == 'group':
            node, level = (node if NAMER.is_family_node(node) else ''), LEVEL_FAMILY
        else:
            node, level = ('' if etype in EXCLUDED_ENTITY_TYPES else node), LEVEL_SPECIFIC
        nodes.append(node)
        levels.append(level)
    return nodes, levels

def load_kinase_features_esm():
    """返回字典 {规范激酶名: ESM特征张量}。.npy 按原始名存储，这里经别名表归一；
    别名合并后多个文件可能落到同一规范名，按 _feature_rank 只保留一个。"""
    best = {}                      # 规范名 -> (rank, 特征)
    for dir_path in KINASE_FEATURE_DIRS:
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith('.npy'):
                continue
            filepath = os.path.join(dir_path, fname)
            data = np.load(filepath, allow_pickle=True).item()
            raw_name = str(data.get('name', os.path.splitext(fname)[0]))
            norm_name = canonical_kinase_name(raw_name, semantics=_name_semantics(raw_name))
            if not norm_name:
                continue
            mean_dict = data.get('mean')
            if mean_dict:
                layer_feats = []
                for layer in FEATURE_LAYERS:
                    if layer in mean_dict:
                        layer_feats.append(mean_dict[layer].flatten())
                if layer_feats:
                    rank = _feature_rank(raw_name, norm_name)
                    if norm_name not in best or rank < best[norm_name][0]:
                        best[norm_name] = (rank, torch.tensor(np.mean(layer_feats, axis=0), dtype=torch.float))
            # 如果没有 mean，尝试其他格式（略，与原逻辑一致）
    return {name: feat for name, (_, feat) in best.items()}

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
        a = canonical_kinase_name(row['preferredName_A'], semantics='symbol')
        b = canonical_kinase_name(row['preferredName_B'], semantics='symbol')
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
    
    # 读取激酶-底物关系，确定每一行的图节点与边层级（见 classify_kinase_rows）
    df_ks = pd.read_csv(KINASE_SUBSTRATE_CSV)
    df_ks['Kinase_raw'] = df_ks['Kinase'].astype(str).apply(normalize_kinase_name)
    df_ks['site_id'] = df_ks['EPSD_ID'] + '_' + df_ks['Position'].astype(str)
    nodes, levels = classify_kinase_rows(df_ks, load_autophos_map(AUTOPHOS_MAP_CSV))
    df_ks['Kinase_norm'], df_ks['level'] = nodes, levels
    excluded = df_ks.loc[df_ks['Kinase_norm'] == '', 'Kinase_raw'].value_counts()
    if len(excluded):
        logging.info(f"KSR 行归属: 剔除 {int(excluded.sum())} 行（{len(excluded)} 个名字——非蛋白激酶、融合蛋白 / "
                     f"注释文字、没有家族定义的泛称、反推不出激酶的自磷酸化）: "
                     + ", ".join(f"{k}({v})" for k, v in excluded.head(15).items())
                     + ("…" if len(excluded) > 15 else ""))
    used = df_ks[df_ks['Kinase_norm'] != ''].reset_index(drop=True)
    logging.info("KSR 行归属（保留）: " + "，".join(
        f"{name} {int((used['level'] == lv).sum())} 行" for lv, name in
        ((LEVEL_SPECIFIC, "特定激酶的注释"), (LEVEL_FAMILY, "家族层面注释"), (LEVEL_AUTOPHOS, "自磷酸化"))))
    family_defs = NAMER.family_nodes()
    all_kinases_from_ks = set(used['Kinase_norm'].unique())
    real_kinases_from_ks = all_kinases_from_ks - set(family_defs)
    
    # 读取 PPI 边，获取所有激酶（规范名）。STRING 的 preferredName 是现用符号语义
    df_ppi = pd.read_csv(PPI_CSV)
    df_ppi['Kinase_A_norm'] = df_ppi['preferredName_A'].astype(str).apply(
        lambda x: canonical_kinase_name(x, semantics='symbol'))
    df_ppi['Kinase_B_norm'] = df_ppi['preferredName_B'].astype(str).apply(
        lambda x: canonical_kinase_name(x, semantics='symbol'))
    df_ppi = df_ppi[(df_ppi['Kinase_A_norm'] != '') & (df_ppi['Kinase_B_norm'] != '')].reset_index(drop=True)
    all_kinases_from_ppi = set(df_ppi['Kinase_A_norm']) | set(df_ppi['Kinase_B_norm'])
    
    # 候选激酶节点 = 出现在关系或 PPI 中的激酶（家族节点在成员特征确定之后再建）
    candidate_kinases = real_kinases_from_ks | all_kinases_from_ppi
    
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

    # 家族节点：承接家族层面的注释，特征取成员特征的质心；没有任何成员有特征的家族节点不进图
    family_feat = {}
    for f in sorted(all_kinases_from_ks):
        if f in family_defs:
            members = [m for m in family_defs[f] if m in kinase_feat]
            if members:
                family_feat[f] = members
                kinase_feat[f] = None                 # 节点是否存在只看键；特征在下面按成员求均值
    logging.info(f"家族节点 {len(family_feat)} 个（FAMILY_xxx，只用于训练）")
    
    # 过滤关系边：只保留激酶在 kinase_feat 中且底物在 site_feat 中的行；
    # 别名合并后同一 (激酶, 位点) 会重复出现（CDC2 与 CDK1 的重叠边、跨库重复条目），去重
    mask_ks = used['Kinase_norm'].isin(kinase_feat.keys()) & used['site_id'].isin(site_feat.keys())
    # 同一 (激酶, 位点) 会重复出现（别名合并后的重叠边、跨库重复条目、自磷酸化与特异注释并存）：
    # 按层级排序后去重，保留 level 最小的那条（特异注释优先于家族层面 / 自磷酸化）
    df_ks_filtered = used[mask_ks].sort_values('level', kind='stable').drop_duplicates(['Kinase_norm', 'site_id'])
    n_lv = df_ks_filtered['level'].value_counts()
    logging.info(f"磷酸化边: 入图 {len(df_ks_filtered)} 条（去重前 {int(mask_ks.sum())} 条）—— "
                 f"特定激酶 {int(n_lv.get(LEVEL_SPECIFIC, 0))}，家族层面 {int(n_lv.get(LEVEL_FAMILY, 0))}，"
                 f"自磷酸化 {int(n_lv.get(LEVEL_AUTOPHOS, 0))}")
    
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

    # 家族节点特征 = 成员特征均值（ESM、Node2Vec 各自对有该特征的成员求均值）；
    # family_members 记录每个家族节点的成员下标，训练脚本据此做负样本过滤与评估屏蔽
    family_members = [[] for _ in all_kinases]
    for i, k in enumerate(all_kinases):
        if k not in family_feat:
            continue
        ms = family_feat[k]
        family_members[i] = [kinase_to_idx[m] for m in ms]
        e = [kinase_esm[m] for m in ms if m in kinase_esm]
        if esm_x is not None and e:
            esm_x[i] = torch.tensor(np.mean(e, axis=0), dtype=torch.float)
        n = [kinase_n2v[m] for m in ms if m in kinase_n2v]
        if n:
            n2v_x[i] = torch.tensor(np.mean(n, axis=0), dtype=torch.float)
    
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
    edge_level = torch.tensor(df_ks_filtered['level'].tolist(), dtype=torch.long)   # 与 edge_ks 同序
    
    # 构建 PPI 双向边：复用前面已按规范名过滤的 df_ppi_filtered；
    # 别名合并后不同行可能落到同一对节点，按无序对去重
    ppi_pairs = {
        tuple(sorted((kinase_to_idx[row['Kinase_A_norm']], kinase_to_idx[row['Kinase_B_norm']])))
        for _, row in df_ppi_filtered.iterrows()
    }
    edge_ppi = []
    for u, v in sorted(ppi_pairs):
        if u != v:
            edge_ppi.append([u, v])
            edge_ppi.append([v, u])
    edge_ppi = torch.tensor(edge_ppi, dtype=torch.long).t().contiguous() if edge_ppi else torch.empty((2, 0), dtype=torch.long)
    
    data['kinase', 'phosphorylates', 'site'].edge_index = edge_ks
    data['site', 'rev_phosphorylates', 'kinase'].edge_index = edge_ks[[1, 0], :]
    data['kinase', 'ppis', 'kinase'].edge_index = edge_ppi
    # 边层级与家族节点标注（反向边与正向边同序）：训练脚本经 data_prep/edge_levels.py 读取
    data['kinase', 'phosphorylates', 'site'].edge_level = edge_level
    data['site', 'rev_phosphorylates', 'kinase'].edge_level = edge_level
    
    # 保存节点名称
    data['kinase'].node_names = all_kinases
    data['site'].node_names = all_sites
    data['kinase'].is_family = torch.tensor([k in family_feat for k in all_kinases], dtype=torch.bool)
    data['kinase'].family_members = family_members
    
    # 9. 保存
    torch.save(data, OUTPUT_GRAPH)
    logging.info(f"联合异构图已保存至 {OUTPUT_GRAPH}")
    logging.info(f"图结构:\n{data}")
    merged = {k: NAMER.aliases(k) for k in all_kinases
              if k not in family_feat and len(NAMER.aliases(k)) > 1}
    logging.info(f"激酶节点 {len(all_kinases)} 个（其中家族节点 {len(family_feat)} 个），"
                 f"由多个名字合并而成的 {len(merged)} 个: "
                 + ", ".join(f"{k}<-{'/'.join(v for v in names if v != k)}" for k, names in sorted(merged.items())))
    fam_edges = df_ks_filtered[df_ks_filtered['level'] == LEVEL_FAMILY].groupby('Kinase_norm').size()
    logging.info("家族节点及其边数: " + ", ".join(
        f"{k}[{'/'.join(family_feat[k])}]x{int(fam_edges.get(k, 0))}" for k in sorted(family_feat)))
    
    return data

if __name__ == "__main__":
    build_joint_graph()