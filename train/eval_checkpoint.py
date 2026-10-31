#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
头对头评估：在完全相同的固定评估协议（同一份 val/test 正负样本）下
对比任意多个 checkpoint，消除"换协议后指标不可比"的问题。

用法（WSL 项目根目录）:
    python3 train/eval_checkpoint.py model/best_joint_model_new.pt [model/xxx.pt ...]

复现 train_hgt_trans.py main() 的划分（RANDOM_SEED=42, cold-start 按激酶划分）
与固定评估样本（seed=RANDOM_SEED+1000+i, num_neg=5, tail 策略），
逐个加载 checkpoint 计算 Val/Test 的 AUC 与 PR-AUC，输出对照表。

注意：仅兼容 P2 之后的模型结构（similar_to 边 + 家族大组层级）。
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_hgt_trans import (
    DEVICE, GRAPH_PATH, KINASE_FAMILY_FILE, RANDOM_SEED, NEG_STRATEGY,
    ESM_KNN_K, ESM_SIM_MIN, NUM_KINASE_GROUPS,
    RETRIEVAL_TOP_M, RETRIEVAL_TAU, RETRIEVAL_SCALE,
    load_kinase_family_labels, build_true_edge_maps, build_fixed_eval_pairs,
    select_best_f1_threshold, add_esm_knn_edges, build_family_groups,
    build_retrieval_profile,
)
from model.hgt_transformer import HGTTransformer


def build_protocol():
    """复现训练脚本的划分与固定评估样本，返回评估所需的全部上下文。"""
    data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
    kinase_names = data['kinase'].node_names
    num_kinase = len(kinase_names)
    num_site = data['site'].x.size(0)
    kinase_labels, num_families = load_kinase_family_labels(kinase_names, KINASE_FAMILY_FILE)
    # P2: 与训练脚本一致地加 ESM-kNN 相似边与家族大组（必须在 clone 前作用于 data）
    add_esm_knn_edges(data, ESM_KNN_K, ESM_SIM_MIN)
    _, family_group_map = build_family_groups(data, kinase_labels, num_families, NUM_KINASE_GROUPS, RANDOM_SEED)

    full_edge_index = data['kinase', 'phosphorylates', 'site'].edge_index
    kinase_sites, site_kinases = build_true_edge_maps(full_edge_index)

    # cold-start 按激酶划分（与训练脚本 SPLIT_MODE="kinase" 完全一致）
    edge_kinases = full_edge_index[0].cpu().numpy()
    kinases_with_edges = np.unique(edge_kinases)
    train_k, temp_k = train_test_split(
        kinases_with_edges, test_size=0.3, random_state=RANDOM_SEED
    )
    val_k, test_k = train_test_split(
        temp_k, test_size=1/3, random_state=RANDOM_SEED
    )
    train_edge_index = full_edge_index[:, torch.as_tensor(
        np.isin(edge_kinases, train_k), device=full_edge_index.device)].to(DEVICE)
    val_edge_index = full_edge_index[:, torch.as_tensor(
        np.isin(edge_kinases, val_k), device=full_edge_index.device)].to(DEVICE)
    test_edge_index = full_edge_index[:, torch.as_tensor(
        np.isin(edge_kinases, test_k), device=full_edge_index.device)].to(DEVICE)

    # R1: 与训练脚本一致地构建检索谱先验（只用训练激酶/训练边，零泄漏）
    train_kinase_mask = torch.as_tensor(np.isin(kinase_names, train_k))
    retrieval_log, _ = build_retrieval_profile(
        data['kinase'].esm_x, train_kinase_mask.to(DEVICE), train_edge_index, num_site,
        top_m=RETRIEVAL_TOP_M, tau=RETRIEVAL_TAU, scale=RETRIEVAL_SCALE,
    )

    # 嵌入只由训练边快照计算（与训练期评估一致，避免评估边泄漏）
    train_data = data.clone()
    train_data['kinase', 'phosphorylates', 'site'].edge_index = train_edge_index
    train_data['site', 'rev_phosphorylates', 'kinase'].edge_index = train_edge_index[[1, 0]]

    # 固定评估样本：与训练脚本相同 seed，保证逐边一致
    fixed_pairs = {}
    for i, (split, eidx) in enumerate((('val', val_edge_index), ('test', test_edge_index))):
        fixed_pairs[split] = build_fixed_eval_pairs(
            eidx, num_neg=5, num_kinase=num_kinase, num_site=num_site,
            kinase_sites=kinase_sites, site_kinases=site_kinases,
            strategy=NEG_STRATEGY, seed=RANDOM_SEED + 1000 + i,
        )

    esm_dim = data['kinase'].esm_x.size(1)
    n2v_dim = data['kinase'].n2v_x.size(1)
    site_dim = data['site'].x.size(1)
    site_window = getattr(data['site'], 'window', None)
    motif_len = int(site_window.size(1)) if site_window is not None else 0

    print(f"协议: cold-start 按激酶划分 | 训练激酶 {len(train_k)}, "
          f"验证激酶 {len(val_k)}({val_edge_index.size(1)} 条边), "
          f"测试激酶 {len(test_k)}({test_edge_index.size(1)} 条边) | "
          f"负采样 {NEG_STRATEGY} x5 | PR-AUC 随机基线 ≈ {1/6:.3f}")
    return (train_data, fixed_pairs, (esm_dim, n2v_dim, site_dim, num_families, motif_len,
            int(family_group_map.size(1))), family_group_map, retrieval_log)


def eval_checkpoint(model, ckpt_path, train_data, fixed_pairs):
    """加载单个 checkpoint，在同一份固定样本上计算 Val/Test AUC、PR-AUC
    与 val 校准阈值的 F1（阈值只从 val 选，test 复用同一阈值）。"""
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    out = {}
    split_preds = {}
    with torch.no_grad():
        emb = model(train_data)
        k_emb, s_emb = emb['kinase'], emb['site']
        for split in ('val', 'test'):
            pos_k, pos_s, neg_k, neg_s = fixed_pairs[split]
            pos_pred = model.predict_interaction(k_emb, s_emb, pos_k, pos_s).cpu().numpy()
            neg_pred = model.predict_interaction(k_emb, s_emb, neg_k, neg_s).cpu().numpy()
            pred = np.concatenate([pos_pred, neg_pred])
            label = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
            out[f'{split}_auc'] = roc_auc_score(label, pred)
            out[f'{split}_pr_auc'] = average_precision_score(label, pred)
            split_preds[split] = (pred, label)
    thr, _ = select_best_f1_threshold(split_preds['val'][1], split_preds['val'][0])
    for split in ('val', 'test'):
        pred, label = split_preds[split]
        out[f'{split}_f1'] = f1_score(label, (pred >= thr).astype(int), zero_division=0)
    out['threshold'] = thr
    return out


def main():
    ckpt_paths = sys.argv[1:]
    if not ckpt_paths:
        print(__doc__)
        sys.exit(1)

    train_data, fixed_pairs, dims, family_group_map, retrieval_log = build_protocol()
    esm_dim, n2v_dim, site_dim, num_families, motif_len, num_groups = dims
    # hidden_dim 从 checkpoint 自动推断（训练侧可能改动，如 256→128）
    state0 = torch.load(ckpt_paths[0], map_location=DEVICE, weights_only=False)
    hidden_dim = int(state0['family_prototypes'].shape[1])
    del state0
    model = HGTTransformer(
        esm_dim=esm_dim, n2v_dim=n2v_dim, site_dim=site_dim,
        num_families=num_families, hidden_dim=hidden_dim, num_heads=4,
        num_hgt_layers=2, num_transformer_layers=1, dropout=0.2,
        motif_len=motif_len,
        num_groups=num_groups, family_group_map=family_group_map,
    ).to(DEVICE)
    # R1: 注入检索谱先验——与训练时一致；漏掉则 R1 分支在评估中静默缺失，指标不可比
    model.retrieval_profile = retrieval_log.to(DEVICE)

    print(f"\n{'checkpoint':<40} {'Val AUC':>8} {'Val PR':>8} {'Val F1*':>8} "
          f"{'Test AUC':>9} {'Test PR':>8} {'Test F1*':>8} {'thr':>7}")
    print("-" * 102)
    for path in ckpt_paths:
        try:
            r = eval_checkpoint(model, path, train_data, fixed_pairs)
        except Exception as e:
            print(f"{Path(path).name:<40} 加载/评估失败: {e}")
            continue
        print(f"{Path(path).name:<40} {r['val_auc']:>8.4f} {r['val_pr_auc']:>8.4f} {r['val_f1']:>8.4f} "
              f"{r['test_auc']:>9.4f} {r['test_pr_auc']:>8.4f} {r['test_f1']:>8.4f} {r['threshold']:>7.4f}")
    print("F1* = 阈值由 val 校准后复用到 test（阈值不接触测试数据）")


if __name__ == "__main__":
    main()
