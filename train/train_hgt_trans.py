#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双任务联合训练：激酶家族分类 + 激酶-底物关系预测
共享 GNN 编码器 (HGT + Transformer)

划分协议（链接预测）：SPLIT_MODE='kinase' 时按激酶整体留出（cold-start），
  验证/测试激酶的全部磷酸化边不进入训练图，模拟暗激酶（无已知底物）的零样本条件；
  SPLIT_MODE='edge' 为旧的按边随机划分（transductive，仅用于对照）。
训练与评估的节点嵌入始终由仅含训练边的图快照(train_data)计算，
  避免验证/测试边泄漏进 HGT 消息传递导致指标虚高。
负采样：对正样本做单侧破坏（默认保留激酶替换位点），并以全量正边索引过滤，
  避免把未标注的真实边（尤其暗激酶的边）当作负样本。
"""

from collections import defaultdict
from pathlib import Path

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, auc, roc_curve, f1_score, precision_score, recall_score, average_precision_score
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# 直接运行 python3 train/train_hgt_trans.py 时 sys.path[0] 是 train/，
# 需先把项目根目录加入 sys.path 才能导入 model 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.hgt_transformer import HGTTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 11

# ==================== 配置参数 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = str(PROJECT_ROOT / "origin_data/joint_hetero_graph_motif.pt")
KINASE_FAMILY_FILE = str(PROJECT_ROOT / "origin_data/kinase_family.xls")   # 列名: Name, Family

LINK_PRED_BATCH_SIZE = 2000      # 每次采样多少正样本对
NUM_NEG_SAMPLES = 5               # 每个正样本对应的负样本数（1:1 会高估 PR-AUC）
SPLIT_MODE = "kinase"             # "kinase": 按激酶整体留出(cold-start) | "edge": 按边随机划分(对照)
NEG_STRATEGY = "tail"             # "tail": 保留激酶替换位点 | "head": 保留位点替换激酶 | "mixed": 各半
LAMBDA_FAMILY = 0.5               # 分类损失权重
LAMBDA_LINK = 1.0                 # 链接预测损失权重
LAMBDA_RANK = 0.5                 # L3 排序损失权重（margin ranking，直接优化排序/AUC）
RANKING_MARGIN = 1.0              # L3 排序损失间隔
EDGE_DROP_RATIO = 0.3             # L1 冷启动模拟：每轮随机屏蔽的训练激酶比例（其磷酸化边只退出消息传递，仍计入损失）
# ---- P1 排序目标改造 ----
HARD_NEG_PROBS = (0.4, 0.3, 0.3)  # 训练负样本策略混合比例: (随机tail, 同家族激酶head, 同蛋白位点tail)
DEGREE_ALPHA = 0.5                # 激酶均衡采样: 正边采样权重 ∝ 1/deg(kinase)^alpha（0=旧均匀采样, 1=完全均衡）
LAMBDA_NCE = 0.5                  # L4 批内对比损失权重（InfoNCE，直接优化排序）
NCE_TEMP = 0.2                    # L4 温度
NCE_BATCH = 256                   # L4 子批大小（B×B 交叉打分，OOM 时调小）
LEARNING_RATE = 8e-5
EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 30

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 加载数据 ====================
def load_kinase_family_labels(kinase_nodes, family_file):
    """
    加载激酶家族标签。
    参数:
        kinase_nodes: list, 所有激酶节点名称（按字母排序，与图中顺序一致）
        family_file: 文件路径（包含 Name, Family 两列）
    返回:
        labels: torch.LongTensor, 长度为 len(kinase_nodes)，没有标签的为 -1（忽略）
        num_classes: int
    """
    df = pd.read_excel(family_file)
    # 归一化激酶名称（与构建图时一致：大写，不删数字）
    df['Kinase_norm'] = df['Name'].astype(str).str.strip().str.upper()
    # 构建映射
    label_map = {}
    for _, row in df.iterrows():
        label_map[row['Kinase_norm']] = row['Family']
    # 编码家族名称
    families = sorted(set(label_map.values()))
    le = LabelEncoder()
    le.fit(families)
    labels = []
    for k in kinase_nodes:
        if k in label_map:
            labels.append(le.transform([label_map[k]])[0])
        else:
            labels.append(-1)  # 缺失标签
    labels = torch.tensor(labels, dtype=torch.long)
    num_classes = len(families)
    logging.info(f"激酶家族类别数: {num_classes}, 有标签的激酶数: {(labels != -1).sum().item()}")
    return labels, num_classes

def build_true_edge_maps(full_edge_index):
    """
    由全量磷酸化边（训练+验证+测试）构建真实邻居索引，用于负样本过滤：
    任何已知正边都不得作为负样本出现。
    返回:
        kinase_sites: dict kinase_idx -> set(site_idx)
        site_kinases: dict site_idx -> set(kinase_idx)
    """
    kinase_sites = defaultdict(set)
    site_kinases = defaultdict(set)
    for k, s in zip(full_edge_index[0].cpu().numpy().tolist(),
                    full_edge_index[1].cpu().numpy().tolist()):
        kinase_sites[k].add(s)
        site_kinases[s].add(k)
    return kinase_sites, site_kinases

def _corrupt_and_filter(fixed_side, num_candidates, true_neighbors):
    """
    保持 fixed_side 不变，为每个元素随机采一个替换侧候选；
    候选若命中该节点在全量正边中的真实邻居则重采样。
    """
    cand = np.random.randint(0, num_candidates, size=len(fixed_side))
    for _ in range(100):
        collide = np.fromiter(
            (int(c) in true_neighbors[int(f)] for f, c in zip(fixed_side, cand)),
            dtype=bool, count=len(fixed_side),
        )
        if not collide.any():
            break
        cand[collide] = np.random.randint(0, num_candidates, size=int(collide.sum()))
    return cand

def _generate_negatives(pos_k, pos_s, num_kinase, num_site,
                        kinase_sites, site_kinases, strategy):
    """
    对一批正样本做单侧破坏，生成等量负样本。
    strategy: "tail" 保留激酶、随机替换位点（评估"给定激酶，哪些位点被磷酸化"）；
              "head" 保留位点、随机替换激酶；"mixed" 两者各半。
    """
    if strategy == "tail":
        neg_k = pos_k.copy()
        neg_s = _corrupt_and_filter(pos_k, num_site, kinase_sites)
    elif strategy == "head":
        neg_s = pos_s.copy()
        neg_k = _corrupt_and_filter(pos_s, num_kinase, site_kinases)
    elif strategy == "mixed":
        neg_k = pos_k.copy()
        neg_s = pos_s.copy()
        half = len(pos_k) // 2
        if half > 0:
            neg_s[:half] = _corrupt_and_filter(pos_k[:half], num_site, kinase_sites)
        if len(pos_k) - half > 0:
            neg_k[half:] = _corrupt_and_filter(pos_s[half:], num_kinase, site_kinases)
    else:
        raise ValueError(f"未知负采样策略: {strategy}")
    return neg_k, neg_s

def _generate_mixed_negatives(pos_k, pos_s, num_kinase, num_site,
                              kinase_sites, site_kinases,
                              kinase_family, family_kinases,
                              site_protein, protein_sites, probs):
    """P1 难负样本混合：每条负样本独立按 probs 抽策略——
    - 随机 tail：保留激酶换随机位点（旧策略，维持基础难度）
    - 同家族 head：保留位点换同家族激酶，逼模型学"家族内哪个成员"而非只学家族偏好
    - 同蛋白 tail：保留激酶换同蛋白其它注释位点，蛋白上下文相同，只靠 motif 区分
    候选不可行（家族唯一成员 / 蛋白仅一个位点 / 候选全是真边）时回退随机破坏。
    """
    n = len(pos_k)
    neg_k = pos_k.copy()
    neg_s = pos_s.copy()
    u = np.random.random(n)
    m_rand = u < probs[0]
    m_fam = (~m_rand) & (u < probs[0] + probs[1])
    m_prot = ~(m_rand | m_fam)

    if m_rand.any():
        neg_s[m_rand] = _corrupt_and_filter(pos_k[m_rand], num_site, kinase_sites)

    # 同家族激酶替换（head 破坏）
    for i in np.where(m_fam)[0]:
        cands = family_kinases.get(int(kinase_family[pos_k[i]]))
        ok = False
        if cands:
            for _ in range(10):
                c = cands[np.random.randint(len(cands))]
                if c != pos_k[i] and c not in site_kinases[int(pos_s[i])]:
                    neg_k[i] = c
                    ok = True
                    break
        if not ok:   # 家族唯一成员或候选均为真边 → 回退随机 head
            neg_k[i] = _corrupt_and_filter(np.array([pos_s[i]]), num_kinase, site_kinases)[0]

    # 同蛋白位点替换（tail 破坏）
    for i in np.where(m_prot)[0]:
        cands = protein_sites.get(site_protein[int(pos_s[i])])
        ok = False
        if cands and len(cands) > 1:
            for _ in range(10):
                c = cands[np.random.randint(len(cands))]
                if c != pos_s[i] and c not in kinase_sites[int(pos_k[i])]:
                    neg_s[i] = c
                    ok = True
                    break
        if not ok:   # 蛋白仅一个注释位点或候选均为真边 → 回退随机 tail
            neg_s[i] = _corrupt_and_filter(np.array([pos_k[i]]), num_site, kinase_sites)[0]
    return neg_k, neg_s

def sample_link_prediction_pairs(edge_index_ks, num_kinase, num_site, num_neg=1,
                                 batch_size=10000, kinase_sites=None, site_kinases=None,
                                 strategy="tail"):
    """
    随机采样正样本对，并按 strategy 生成 num_neg 倍的单侧破坏负样本
    （负样本与全量正边索引比对过滤，不与任何已知激酶-位点对重合）。
    返回的索引张量在 edge_index_ks 所在设备上。
    """
    if kinase_sites is None or site_kinases is None:
        raise ValueError("需传入全量正边索引 kinase_sites/site_kinases 用于负样本过滤")
    device = edge_index_ks.device
    pos_kinase = edge_index_ks[0].cpu().numpy()
    pos_site = edge_index_ks[1].cpu().numpy()
    total_pos = len(pos_kinase)

    if batch_size > total_pos:
        indices = np.arange(total_pos)
    else:
        indices = np.random.choice(total_pos, batch_size, replace=False)
    pos_k_np = pos_kinase[indices]
    pos_s_np = pos_site[indices]

    neg_k_list, neg_s_list = [], []
    for _ in range(num_neg):
        nk, ns = _generate_negatives(pos_k_np, pos_s_np, num_kinase, num_site,
                                     kinase_sites, site_kinases, strategy)
        neg_k_list.append(nk)
        neg_s_list.append(ns)

    pos_k = torch.tensor(pos_k_np, dtype=torch.long, device=device)
    pos_s = torch.tensor(pos_s_np, dtype=torch.long, device=device)
    neg_k = torch.tensor(np.concatenate(neg_k_list), dtype=torch.long, device=device)
    neg_s = torch.tensor(np.concatenate(neg_s_list), dtype=torch.long, device=device)
    return pos_k, pos_s, neg_k, neg_s

def sample_train_pairs(train_edge_index, batch_size, num_neg, num_kinase, num_site,
                       kinase_sites, site_kinases, hard_ctx, edge_weights=None):
    """P1 训练对采样：正边按激酶度反加权（稀有激酶过采样，DEGREE_ALPHA 控制强度），
    负样本为难负混合（HARD_NEG_PROBS）。负样本布局与 sample_link_prediction_pairs
    一致（round-major：第 r 轮的第 i 条负样本位于 r*n_pos+i）。"""
    device = train_edge_index.device
    pos_k_all = train_edge_index[0].cpu().numpy()
    pos_s_all = train_edge_index[1].cpu().numpy()
    total = len(pos_k_all)
    if edge_weights is None:
        indices = np.random.choice(total, batch_size, replace=total < batch_size)
    else:
        indices = np.random.choice(total, batch_size, replace=True, p=edge_weights)
    pos_k_np = pos_k_all[indices]
    pos_s_np = pos_s_all[indices]

    neg_k_list, neg_s_list = [], []
    for _ in range(num_neg):
        nk, ns = _generate_mixed_negatives(
            pos_k_np, pos_s_np, num_kinase, num_site,
            kinase_sites, site_kinases,
            hard_ctx['kinase_family'], hard_ctx['family_kinases'],
            hard_ctx['site_protein'], hard_ctx['protein_sites'], HARD_NEG_PROBS,
        )
        neg_k_list.append(nk)
        neg_s_list.append(ns)

    pos_k = torch.tensor(pos_k_np, dtype=torch.long, device=device)
    pos_s = torch.tensor(pos_s_np, dtype=torch.long, device=device)
    neg_k = torch.tensor(np.concatenate(neg_k_list), dtype=torch.long, device=device)
    neg_s = torch.tensor(np.concatenate(neg_s_list), dtype=torch.long, device=device)
    return pos_k, pos_s, neg_k, neg_s

def build_fixed_eval_pairs(edge_index, num_neg, num_kinase, num_site,
                           kinase_sites, site_kinases, strategy, seed):
    """
    为一个评估边集预生成固定的正负样本对（消除评估噪声）：
    - 正样本：该边集的全部边（不再随机抽样子集）
    - 负样本：按 strategy 做 num_neg 次单侧破坏，经全量正边过滤
    用独立 seed 生成后恢复全局随机状态，不影响训练期随机性。
    返回: (pos_k, pos_s, neg_k, neg_s)，供 evaluate / get_link_pred_predictions 复用。
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        device = edge_index.device
        pos_k_np = edge_index[0].cpu().numpy()
        pos_s_np = edge_index[1].cpu().numpy()
        neg_k_list, neg_s_list = [], []
        for _ in range(num_neg):
            nk, ns = _generate_negatives(pos_k_np, pos_s_np, num_kinase, num_site,
                                         kinase_sites, site_kinases, strategy)
            neg_k_list.append(nk)
            neg_s_list.append(ns)
        pos_k = torch.tensor(pos_k_np, dtype=torch.long, device=device)
        pos_s = torch.tensor(pos_s_np, dtype=torch.long, device=device)
        neg_k = torch.tensor(np.concatenate(neg_k_list), dtype=torch.long, device=device)
        neg_s = torch.tensor(np.concatenate(neg_s_list), dtype=torch.long, device=device)
    finally:
        np.random.set_state(rng_state)
    return pos_k, pos_s, neg_k, neg_s

# ==================== 训练和评估函数（支持掩码）====================
def train_epoch(model, data, kinase_labels, train_kinase_mask, train_edge_index,
                optimizer, criterion_class, criterion_link, kinase_sites, site_kinases,
                hard_ctx=None, edge_weights=None):
    model.train()
    optimizer.zero_grad()

    # L1 冷启动模拟：随机选一部分训练激酶，把它们的磷酸化边（正向+反向）
    # 从消息传递图快照中临时移除（PPI 边保留）。这些边仍参与下面的链接损失，
    # 迫使模型仅凭输入特征（ESM/n2v/motif）为这些激酶打分——即验证/测试的推理模式
    phos_key = ('kinase', 'phosphorylates', 'site')
    rev_key = ('site', 'rev_phosphorylates', 'kinase')
    orig_phos_ei = data[phos_key].edge_index
    orig_rev_ei = data[rev_key].edge_index if rev_key in data.edge_types else None
    if EDGE_DROP_RATIO > 0:
        num_k = data['kinase'].esm_x.size(0)
        drop_k = torch.randperm(num_k, device=orig_phos_ei.device)[:int(EDGE_DROP_RATIO * num_k)]
        data[phos_key].edge_index = orig_phos_ei[:, ~torch.isin(orig_phos_ei[0], drop_k)]
        if orig_rev_ei is not None:
            data[rev_key].edge_index = orig_rev_ei[:, ~torch.isin(orig_rev_ei[1], drop_k)]

    emb_dict = model(data)   # {'kinase': [K, H], 'site': [S, H]}
    data[phos_key].edge_index = orig_phos_ei   # 立即恢复，保证评估用的图快照不受影响
    if orig_rev_ei is not None:
        data[rev_key].edge_index = orig_rev_ei
    kinase_emb = emb_dict['kinase']
    site_emb = emb_dict['site']

    # 1. 激酶家族分类损失（仅对训练集中的有标签激酶）
    family_logits = model.predict_family(kinase_emb)   # [K, num_families]
    fam_probs = torch.softmax(family_logits, dim=-1)   # 复用给 A1 家族原型先验分支
    mask = (kinase_labels != -1) & train_kinase_mask
    if mask.sum() > 0:
        loss_family = criterion_class(family_logits[mask], kinase_labels[mask])
    else:
        loss_family = torch.tensor(0.0, device=DEVICE)

    # 2. 链接预测损失（训练边 + 单侧破坏负样本，负样本经全量正边过滤）
    num_kinase = kinase_emb.size(0)
    num_site = site_emb.size(0)
    # P1: 训练对改为 度反加权正采样 + 难负混合（评估协议不变，仍用随机 tail 固定样本）
    pos_k, pos_s, neg_k, neg_s = sample_train_pairs(
        train_edge_index, LINK_PRED_BATCH_SIZE, NUM_NEG_SAMPLES, num_kinase, num_site,
        kinase_sites, site_kinases, hard_ctx=hard_ctx, edge_weights=edge_weights,
    )
    pos_logits = model.interaction_logits(kinase_emb, site_emb, pos_k, pos_s, fam_probs=fam_probs)
    neg_logits = model.interaction_logits(kinase_emb, site_emb, neg_k, neg_s, fam_probs=fam_probs)
    pred = torch.cat([pos_logits, neg_logits], dim=0)
    label = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]).float()
    loss_link = criterion_link(pred, label)

    # 3. L3 排序损失：正样本 logit 应比其同激酶配对的负样本 logit 高出至少 margin。
    # 负样本为 round-major 布局：第 r 轮的第 i 条负样本位于 r*n_pos+i
    n_pos = pos_logits.size(0)
    neg_sel_idx = (
        torch.randint(0, NUM_NEG_SAMPLES, (n_pos,), device=pos_logits.device) * n_pos
        + torch.arange(n_pos, device=pos_logits.device)
    )
    loss_rank = F.relu(RANKING_MARGIN - pos_logits + neg_logits[neg_sel_idx]).mean()

    # 4. L4 批内对比损失（InfoNCE）：子批内所有 (k_i, s_j) 交叉打分，对角线为正，
    # 非对角的已知真边从分母剔除；双向 CE 直接优化"真底物排在其它位点前"——与 PR-AUC 语义对齐
    loss_nce = torch.tensor(0.0, device=pos_logits.device)
    if LAMBDA_NCE > 0 and n_pos >= 2 and hard_ctx is not None:
        m = min(NCE_BATCH, n_pos)
        sel = torch.randperm(n_pos, device=pos_logits.device)[:m]
        bk, bs = pos_k[sel], pos_s[sel]
        ki = bk.repeat_interleave(m)
        sj = bs.repeat(m)
        pair_logits = model.interaction_logits(
            kinase_emb, site_emb, ki, sj, fam_probs=fam_probs
        ).view(m, m)
        # 剔除非对角的已知真边（不作为负样本参与分母）
        with torch.no_grad():
            keys = (ki * num_site + sj).cpu().numpy()
            is_true = np.isin(keys, hard_ctx['true_edge_keys'])
            eye = np.eye(m, dtype=bool)
            block = torch.tensor(is_true.reshape(m, m) & ~eye, device=pair_logits.device)
        pair_logits = pair_logits.masked_fill(block, float('-inf'))
        tgt = torch.arange(m, device=pair_logits.device)
        loss_nce = 0.5 * (
            F.cross_entropy(pair_logits / NCE_TEMP, tgt)
            + F.cross_entropy(pair_logits.t() / NCE_TEMP, tgt)
        )

    loss = (LAMBDA_FAMILY * loss_family + LAMBDA_LINK * loss_link
            + LAMBDA_RANK * loss_rank + LAMBDA_NCE * loss_nce)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, embed_data, kinase_labels, mask_dict, edge_index_dict,
             kinase_sites, site_kinases, num_neg_eval=5, neg_strategy="tail",
             fixed_pairs=None):
    """
    评估分类和链接预测性能。
    embed_data: 计算节点嵌入所用的图快照，必须只含训练边，
                否则验证/测试边会泄漏进消息传递导致指标虚高。
    mask_dict: {'train': train_kinase_mask, 'val': val_kinase_mask, 'test': test_kinase_mask}
    edge_index_dict: {'train': train_edge_index, 'val': val_edge_index, 'test': test_edge_index}
    fixed_pairs: 可选 {split: (pos_k, pos_s, neg_k, neg_s)}，预生成的固定评估样本；
                 提供的 split 复用同一份样本（不再随机重采），指标跨轮可比
    """
    model.eval()
    with torch.no_grad():
        emb_dict = model(embed_data)
        kinase_emb = emb_dict['kinase']
        site_emb = emb_dict['site']

        # 分类性能（对每个子集评估）
        family_logits = model.predict_family(kinase_emb)
        results = {}
        for split, mask in mask_dict.items():
            valid_mask = (kinase_labels != -1) & mask
            if valid_mask.sum() > 0:
                pred_class = family_logits[valid_mask].argmax(dim=1)
                acc = (pred_class == kinase_labels[valid_mask]).float().mean().item()
            else:
                acc = 0.0
            results[f'{split}_acc'] = acc

        # 链接预测 AUC（对每个边集采样评估；负样本与训练使用同一策略和过滤规则）
        num_kinase = kinase_emb.size(0)
        num_site = site_emb.size(0)
        for split, eidx in edge_index_dict.items():
            if eidx.size(1) == 0:
                for key in ('auc', 'pr_auc', 'f1', 'acc_link', 'precision', 'recall'):
                    results[f'{split}_{key}'] = 0.0
                continue
            if fixed_pairs is not None and split in fixed_pairs:
                pos_k, pos_s, neg_k, neg_s = fixed_pairs[split]
            else:
                pos_k, pos_s, neg_k, neg_s = sample_link_prediction_pairs(
                    eidx, num_kinase, num_site, num_neg=num_neg_eval,
                    batch_size=min(1000, eidx.size(1)),
                    kinase_sites=kinase_sites, site_kinases=site_kinases,
                    strategy=neg_strategy,
                )
            pos_pred = model.predict_interaction(kinase_emb, site_emb, pos_k, pos_s)
            neg_pred = model.predict_interaction(kinase_emb, site_emb, neg_k, neg_s)
            pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
            label = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
            auc = roc_auc_score(label, pred)
            pr_auc = average_precision_score(label, pred)
            pred_labels = (pred >= 0.5).astype(int)

            results[f'{split}_auc'] = auc
            results[f'{split}_pr_auc'] = pr_auc
            results[f'{split}_f1'] = f1_score(label, pred_labels, zero_division=0)
            results[f'{split}_acc_link'] = accuracy_score(label, pred_labels)
            results[f'{split}_precision'] = precision_score(label, pred_labels, zero_division=0)
            results[f'{split}_recall'] = recall_score(label, pred_labels, zero_division=0)
    return results

def get_link_pred_predictions(model, embed_data, edge_index, kinase_sites, site_kinases,
                              num_neg=10, strategy="tail", fixed_pairs=None):
    """
    在给定的边集上进行链接预测，返回所有正负样本的预测概率和真实标签。
    嵌入由 embed_data（仅含训练边的图快照）计算；
    负样本按 strategy 对每个正样本做 num_neg 次单侧破坏，并经全量正边过滤。
    参数:
        model: 训练好的模型
        embed_data: 仅含训练边的图快照
        edge_index: 待评估的边索引 (2, E)
        num_neg: 每个正样本对应的负样本数 (建议设为 5~10 以获得稳定ROC)
        fixed_pairs: 可选 (pos_k, pos_s, neg_k, neg_s)，提供时跳过内部采样
                     （num_neg/strategy 被忽略），与 evaluate 共用同一份固定样本
    返回:
        preds: numpy array, 预测概率 (长度 E + E*num_neg)
        labels: numpy array, 真实标签 (1为正，0为负)
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        emb_dict = model(embed_data)
        kinase_emb = emb_dict['kinase']
        site_emb = emb_dict['site']
        num_kinase, num_site = kinase_emb.size(0), site_emb.size(0)

        if fixed_pairs is not None:
            pos_k_t, pos_s_t, neg_k_t, neg_s_t = fixed_pairs
        else:
            pos_k = edge_index[0].cpu().numpy()
            pos_s = edge_index[1].cpu().numpy()
            pos_k_t = torch.tensor(pos_k, device=device, dtype=torch.long)
            pos_s_t = torch.tensor(pos_s, device=device, dtype=torch.long)

            neg_k_list, neg_s_list = [], []
            for _ in range(num_neg):
                nk, ns = _generate_negatives(pos_k, pos_s, num_kinase, num_site,
                                             kinase_sites, site_kinases, strategy)
                neg_k_list.append(nk)
                neg_s_list.append(ns)
            neg_k = np.concatenate(neg_k_list)
            neg_s = np.concatenate(neg_s_list)
            neg_k_t = torch.tensor(neg_k, device=device, dtype=torch.long)
            neg_s_t = torch.tensor(neg_s, device=device, dtype=torch.long)

        pos_pred = model.predict_interaction(kinase_emb, site_emb, pos_k_t, pos_s_t).cpu().numpy()
        neg_pred = model.predict_interaction(kinase_emb, site_emb, neg_k_t, neg_s_t).cpu().numpy()

        preds = np.concatenate([pos_pred, neg_pred])
        labels = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    return preds, labels
    
# ==================== 主程序 ====================
def main():
    # 1. 加载图数据
    logging.info("加载异构图...")
    data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
    
    kinase_names = data['kinase'].node_names
    num_kinase = len(kinase_names)
    num_site = data['site'].x.size(0)
    logging.info(f"激酶节点数: {num_kinase}, 底物节点数: {num_site}")
    
    # 2. 加载激酶家族标签
    kinase_labels, num_families = load_kinase_family_labels(kinase_names, KINASE_FAMILY_FILE)
    kinase_labels = kinase_labels.to(DEVICE)
    
    # 3. 获取模型输入维度
    esm_dim = data['kinase'].esm_x.size(1)
    n2v_dim = data['kinase'].n2v_x.size(1)
    site_dim = data['site'].x.size(1)
    
    # 4. 划分数据集（7:2:1）
    # ---- 激酶节点划分（分类任务）----
    # 只对有标签的节点进行随机划分（不采用分层，避免稀有类别报错）
    labeled_indices = torch.where(kinase_labels != -1)[0].cpu().numpy()
    train_idx, temp_idx = train_test_split(
        labeled_indices, test_size=0.3, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1/3, random_state=RANDOM_SEED
    )
    # 构建掩码（全节点长度）
    train_kinase_mask = torch.zeros(num_kinase, dtype=torch.bool)
    val_kinase_mask = torch.zeros(num_kinase, dtype=torch.bool)
    test_kinase_mask = torch.zeros(num_kinase, dtype=torch.bool)
    train_kinase_mask[train_idx] = True
    val_kinase_mask[val_idx] = True
    test_kinase_mask[test_idx] = True
    logging.info(f"激酶分类 - 训练: {train_idx.size}, 验证: {val_idx.size}, 测试: {test_idx.size}")
    
    # ---- 边划分（链接预测任务）----
    full_edge_index = data['kinase', 'phosphorylates', 'site'].edge_index
    # 全量正边索引（训练+验证+测试），负样本采样时据此过滤
    kinase_sites, site_kinases = build_true_edge_maps(full_edge_index)

    if SPLIT_MODE == "kinase":
        # cold-start 划分：每个激酶的全部边进入同一子集，
        # 验证/测试激酶在训练图中没有任何磷酸化边（暗激酶零样本条件）
        edge_kinases = full_edge_index[0].cpu().numpy()
        kinases_with_edges = np.unique(edge_kinases)
        train_k, temp_k = train_test_split(
            kinases_with_edges, test_size=0.3, random_state=RANDOM_SEED
        )
        val_k, test_k = train_test_split(
            temp_k, test_size=1/3, random_state=RANDOM_SEED
        )
        train_edge_index = full_edge_index[:, torch.as_tensor(np.isin(edge_kinases, train_k), device=full_edge_index.device)]
        val_edge_index = full_edge_index[:, torch.as_tensor(np.isin(edge_kinases, val_k), device=full_edge_index.device)]
        test_edge_index = full_edge_index[:, torch.as_tensor(np.isin(edge_kinases, test_k), device=full_edge_index.device)]
        logging.info(
            f"链接预测 - 按激酶划分(cold-start): 训练激酶 {len(train_k)}, 验证激酶 {len(val_k)}, 测试激酶 {len(test_k)} | "
            f"训练边: {train_edge_index.size(1)}, 验证边: {val_edge_index.size(1)}, 测试边: {test_edge_index.size(1)}"
        )
    else:
        # 旧协议：按边随机划分（transductive，仅用于对照）
        edge_indices = np.arange(full_edge_index.size(1))
        train_edge_idx, temp_edge_idx = train_test_split(
            edge_indices, test_size=0.3, random_state=RANDOM_SEED
        )
        val_edge_idx, test_edge_idx = train_test_split(
            temp_edge_idx, test_size=1/3, random_state=RANDOM_SEED
        )
        train_edge_index = full_edge_index[:, train_edge_idx]
        val_edge_index = full_edge_index[:, val_edge_idx]
        test_edge_index = full_edge_index[:, test_edge_idx]
        logging.info(f"链接预测 - 按边划分: 训练边: {train_edge_index.size(1)}, 验证边: {val_edge_index.size(1)}, 测试边: {test_edge_index.size(1)}")
    
    train_data = data.clone()
    train_data['kinase', 'phosphorylates', 'site'].edge_index = train_edge_index
    train_data['site', 'rev_phosphorylates', 'kinase'].edge_index = train_edge_index[[1,0]]

    # 将划分掩码和边索引移到设备
    train_kinase_mask = train_kinase_mask.to(DEVICE)
    val_kinase_mask = val_kinase_mask.to(DEVICE)
    test_kinase_mask = test_kinase_mask.to(DEVICE)
    train_edge_index = train_edge_index.to(DEVICE)
    val_edge_index = val_edge_index.to(DEVICE)
    test_edge_index = test_edge_index.to(DEVICE)
    
    # 5. 模型初始化（若图中位点带有 window 特征，则启用 motif 编码器）
    site_window = getattr(data['site'], 'window', None)
    motif_len = int(site_window.size(1)) if site_window is not None else 0
    logging.info(f"位点 motif 特征: {'启用, 窗口长度 ' + str(motif_len) if motif_len else '未启用'}")
    model = HGTTransformer(
        esm_dim=esm_dim,
        n2v_dim=n2v_dim,
        site_dim=site_dim,
        num_families=num_families,
        hidden_dim=256,
        num_heads=4,
        num_hgt_layers=2,
        num_transformer_layers=1,
        dropout=0.4,
        motif_len=motif_len
    ).to(DEVICE)
    
    # 6. 损失函数与优化器
    criterion_class = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_link = nn.BCEWithLogitsLoss()   # 配合 interaction_logits，数值上比 BCELoss(sigmoid) 更稳定
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 7. 训练循环
    best_val_auc = 0.0
    patience_counter = 0
    
    # 准备掩码字典和边字典供评估使用
    mask_dict = {'train': train_kinase_mask, 'val': val_kinase_mask, 'test': test_kinase_mask}
    edge_dict = {'train': train_edge_index, 'val': val_edge_index, 'test': test_edge_index}

    # P1: 难负样本与均衡采样所需的上下文（一次构建，训练循环复用）
    kinase_family_np = kinase_labels.cpu().numpy()
    family_kinases = defaultdict(list)
    for idx, lab in enumerate(kinase_family_np):
        if lab != -1:
            family_kinases[int(lab)].append(idx)
    site_protein = {}
    protein_sites = defaultdict(list)
    for idx, name in enumerate(data['site'].node_names):
        prot = str(name).rsplit('_', 1)[0]   # 位点名形如 'EP0021246_S298'，前缀为蛋白 EPSD ID
        site_protein[idx] = prot
        protein_sites[prot].append(idx)
    # 全量真边键（k*num_site+s，排序数组，供 L4 批内负样本 O(log n) 过滤）
    _fk = full_edge_index.cpu().numpy()
    true_edge_keys = np.unique(_fk[0].astype(np.int64) * num_site + _fk[1])
    # 正边采样权重 ∝ 1/deg_train(kinase)^DEGREE_ALPHA（稀有激酶过采样）
    _ek = train_edge_index[0].cpu().numpy()
    train_deg = np.bincount(_ek, minlength=num_kinase).astype(np.float64)
    _w = 1.0 / np.maximum(train_deg[_ek], 1.0) ** DEGREE_ALPHA
    train_edge_weights = _w / _w.sum()
    hard_ctx = {
        'kinase_family': kinase_family_np,
        'family_kinases': dict(family_kinases),
        'site_protein': site_protein,
        'protein_sites': dict(protein_sites),
        'true_edge_keys': true_edge_keys,
    }
    logging.info(
        f"P1 难负上下文: 有家族标签激酶 {sum(len(v) for v in family_kinases.values())}, "
        f"蛋白 {len(protein_sites)} 个（多位点蛋白 {sum(1 for v in protein_sites.values() if len(v) > 1)} 个）"
    )

    # 固定评估负样本：val/test 各预生成一份（全部正边 + 固定负样本），每次评估复用，
    # 消除随机重采导致的指标抖动；独立 seed + 状态恢复，不影响训练期随机性
    fixed_eval_pairs = {}
    for i, split in enumerate(('val', 'test')):
        fixed_eval_pairs[split] = build_fixed_eval_pairs(
            edge_dict[split], num_neg=5, num_kinase=num_kinase, num_site=num_site,
            kinase_sites=kinase_sites, site_kinases=site_kinases,
            strategy=NEG_STRATEGY, seed=RANDOM_SEED + 1000 + i,
        )
    
    # logging.info("start training...")
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(
            model, train_data, kinase_labels, train_kinase_mask, train_edge_index,
            optimizer, criterion_class, criterion_link, kinase_sites, site_kinases,
            hard_ctx=hard_ctx, edge_weights=train_edge_weights
        )
        
        # 每 5 个 epoch 评估一次（嵌入由仅含训练边的 train_data 计算，避免评估边泄漏）
        if epoch % 5 == 0:
            results = evaluate(
                model, train_data, kinase_labels, mask_dict, edge_dict,
                kinase_sites, site_kinases, num_neg_eval=5, neg_strategy=NEG_STRATEGY,
                fixed_pairs=fixed_eval_pairs
            )
            val_auc = results['val_auc']
            val_pr_auc = results['val_pr_auc']
            logging.info(
                f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                f"Train Acc: {results['train_acc']:.4f}, Val Acc: {results['val_acc']:.4f} | "
                f"Train AUC: {results['train_auc']:.4f}, Val AUC: {val_auc:.4f}, Val PR-AUC: {results['val_pr_auc']:.4f} | "
                f"Val F1: {results['val_f1']:.4f}, Val Prec: {results['val_precision']:.4f}, Val Rec: {results['val_recall']:.4f}"
            )
            
            # 早停（基于验证集 AUC）
            if val_pr_auc > best_val_auc:
                best_val_auc = val_pr_auc
                patience_counter = 0
                torch.save(model.state_dict(), str(PROJECT_ROOT / "model/best_joint_model_new.pt"))
                logging.info(f"  -> 保存最佳模型 (Val PR-AUC = {val_pr_auc:.4f}, Val F1 = {results['val_f1']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 1 == 0:
                logging.info(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # 加载最佳模型并在测试集上评估
    logging.info(f"训练完成，最佳验证集 PR-AUC: {best_val_auc:.4f}")
    model.load_state_dict(torch.load(str(PROJECT_ROOT / "model/best_joint_model_new.pt"), map_location=DEVICE))
    final_results = evaluate(
        model, train_data, kinase_labels, mask_dict, edge_dict,
        kinase_sites, site_kinases, num_neg_eval=5, neg_strategy=NEG_STRATEGY,
        fixed_pairs=fixed_eval_pairs
    )
    logging.info(
        f"最终测试集性能: "
        f"Link AUC = {final_results['test_auc']:.4f}, "
        f"PR-AUC = {final_results['test_pr_auc']:.4f}, "
        f"Link F1 = {final_results['test_f1']:.4f}, "
        f"Link Acc = {final_results['test_acc_link']:.4f}, "
        f"Precision = {final_results['test_precision']:.4f}, "
        f"Recall = {final_results['test_recall']:.4f}"
    )

    # 获取训练、验证、测试集的预测结果（嵌入统一由 train_data 计算）
    train_preds, train_labels = get_link_pred_predictions(
        model, train_data, train_edge_index, kinase_sites, site_kinases, num_neg=10, strategy=NEG_STRATEGY)
    val_preds, val_labels = get_link_pred_predictions(
        model, train_data, val_edge_index, kinase_sites, site_kinases,
        num_neg=10, strategy=NEG_STRATEGY, fixed_pairs=fixed_eval_pairs['val'])
    test_preds, test_labels = get_link_pred_predictions(
        model, train_data, test_edge_index, kinase_sites, site_kinases,
        num_neg=10, strategy=NEG_STRATEGY, fixed_pairs=fixed_eval_pairs['test'])

    # 计算 ROC 曲线和 AUC
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_preds)
    auc_train = auc(fpr_train, tpr_train)

    fpr_val, tpr_val, _ = roc_curve(val_labels, val_preds)
    auc_val = auc(fpr_val, tpr_val)

    fpr_test, tpr_test, _ = roc_curve(test_labels, test_preds)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(5.5, 5))   # 适合双栏排版

    # Nature 推荐配色
    colors = {'train': '#F4898B', 'val': '#87BFD6', 'test': '#61DCB8'}  # 红、蓝、深蓝

    # 绘制三条曲线
    plt.plot(fpr_train, tpr_train, color=colors['train'], lw=1.2,
            label=f'Train (AUC = {auc_train:.3f})')
    plt.plot(fpr_val, tpr_val, color=colors['val'], lw=1.2,
            label=f'Val (AUC = {auc_val:.3f})')
    plt.plot(fpr_test, tpr_test, color=colors['test'], lw=1.2,
            label=f'Test (AUC = {auc_test:.3f})')

    # 对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random')

    # 坐标轴设置
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # 图例（无边框）
    plt.legend(loc='lower right', frameon=False)

    # 去掉上、右脊线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 保存高分辨率矢量图
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / 'figures/roc_curves.png'), dpi=500, bbox_inches='tight')  # 可选
    plt.show()

if __name__ == "__main__":
    main()