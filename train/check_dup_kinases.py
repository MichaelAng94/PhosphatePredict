#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据质量诊断：重复/近零 ESM 嵌入检查（cold-start 协议完整性）。

P2-B1 日志显示候选相似度 max=1.000（存在完全相同的嵌入——疑似同一激酶
以不同名字重复入库）与 min=0.000（疑似缺失嵌入的零填充）。若"重复身份"对
恰好跨 train/test 划分，cold-start 协议会泄漏，历史指标虚高。

用法（WSL 项目根目录）:
    python3 train/check_dup_kinases.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_hgt_trans import GRAPH_PATH, RANDOM_SEED


def main():
    data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
    names = list(data['kinase'].node_names)
    esm = data['kinase'].esm_x
    num_kinase = len(names)
    print(f"激酶总数 {num_kinase}，ESM 维度 {esm.size(1)}")

    # [1] 近零范数嵌入（kNN 边与检索谱均不可靠）
    norms = esm.norm(dim=1)
    zero_idx = torch.where(norms < 1e-6)[0]
    if len(zero_idx):
        print(f"\n[1] 近零范数嵌入 {len(zero_idx)} 个:")
        for i in zero_idx.tolist():
            print(f"    {names[i]:<16} norm={norms[i]:.2e}")
    else:
        print("\n[1] 无近零范数嵌入")

    # [2] 近重复嵌入对 + cold-start 归属检查
    esm_n = F.normalize(esm, dim=-1)
    sim = esm_n @ esm_n.t()
    sim.fill_diagonal_(0.0)
    ei, ej = torch.where(torch.triu(sim > 0.999, diagonal=1))
    if len(ei):
        # 复现训练脚本的 cold-start 划分（按激酶索引）
        full_edge_index = data['kinase', 'phosphorylates', 'site'].edge_index
        kinases_with_edges = np.unique(full_edge_index[0].cpu().numpy())
        train_k, temp_k = train_test_split(
            kinases_with_edges, test_size=0.3, random_state=RANDOM_SEED)
        val_k, test_k = train_test_split(
            temp_k, test_size=1/3, random_state=RANDOM_SEED)
        tr, vl, te = set(train_k.tolist()), set(val_k.tolist()), set(test_k.tolist())

        def role(i):
            if i in tr:
                return 'train'
            if i in vl:
                return 'val'
            if i in te:
                return 'test'
            return 'no-edge'

        print(f"\n[2] 近重复嵌入对 {len(ei)} 对 (sim>0.999):")
        cross = 0
        for a, b in zip(ei.tolist(), ej.tolist()):
            ra, rb = role(a), role(b)
            flag = ''
            # 双方都有边且分属不同 split → 同一激酶以两个身份分别出现在训练与评估
            if ra != rb and ra != 'no-edge' and rb != 'no-edge':
                flag = '  ←← 跨 split！cold-start 泄漏'
                cross += 1
            print(f"    {names[a]:<14}({ra:<7}) — {names[b]:<14}({rb:<7}) "
                  f"sim={sim[a, b]:.6f}{flag}")
        print(f"    跨 split 重复对: {cross}"
              + ("（>0 则历史指标含泄漏成分，需合并身份重新划分/重建图）" if cross else ""))
    else:
        print("\n[2] 无近重复嵌入对")

    # [3] 相似度分布（R1.1 conf 重新定参参考）
    s = sim[sim > 0]
    if len(s):
        print(f"\n[3] 非零相似度分位数: p10={s.quantile(0.1):.4f} p50={s.quantile(0.5):.4f} "
              f"p90={s.quantile(0.9):.4f} p99={s.quantile(0.99):.4f} max={s.max():.6f}")


if __name__ == '__main__':
    main()
