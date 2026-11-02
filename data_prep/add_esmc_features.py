#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为异构图的位点节点添加 ESMC 窗口嵌入（L1: mean-pooled 向量接入）。

输入:
    1. 图: origin_data/joint_hetero_graph_motif.pt（已有 data['site'].window）
    2. 窗口 FASTA: origin_data/all_windows.fasta（all_window_extractor.py 输出，
       header '>EPSD_ID_STYPOS'，窗口 31aa，'-' 为边界填充）
    3. ESMC 嵌入目录: origin_data/substate/out/（esmc_extractor.py 输出，
       每个窗口一个 .npy: {'name', 'mean': {layer: [d]}}，此处取各层平均）

匹配策略（应对 all_window_extractor 的按序列去重——同序列窗口只保留首个位点）:
    a. 位点名直配: 图位点 'EP0021246_S298' <-> 'EP0021246_S298.npy'
    b. 序列匹配: 图位点的整数 window 反解为序列串（0->'-'），与 all_windows.fasta
       的窗口序列归一化对齐，借用同序列代表位点的嵌入
    c. 仍未匹配: 用已匹配位点的均值填充（缺失数打进日志，>5% 时建议检查上游）

输出: data['site'].esmc_x [num_site, d] float32，per-dim z-score 标准化，
       保存为**新图文件**（默认 joint_hetero_graph_motif_esmc.pt，原输入图不动，
       作为无 ESMC 版备份）；幂等: 输入图已带 esmc_x 则跳过，--force 覆盖。

用法（WSL 项目根目录）:
    python3 data_prep/add_esmc_features.py \
        --graph origin_data/joint_hetero_graph_motif.pt \
        --fasta origin_data/all_windows.fasta \
        --emb-dir origin_data/substate/out/ \
        --out origin_data/joint_hetero_graph_motif_esmc.pt
"""

import argparse
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
ID_TO_AA = {i + 1: aa for i, aa in enumerate(AA_ORDER)}


def window_to_seq(window_row) -> str:
    """整数窗口 [L] 反解为字符串；0（填充/非标准残基）→ '-'。"""
    return ''.join(ID_TO_AA.get(int(v), '-') for v in window_row)


def norm_seq(s: str) -> str:
    """统一大小写并把非标准氨基酸字符归一为 '-'（与 window 反解对齐）。"""
    return ''.join(c if c in AA_ORDER else '-' for c in s.upper())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default='origin_data/joint_hetero_graph_motif.pt',
                    help='输入图（无 ESMC 特征）')
    ap.add_argument('--fasta', default='origin_data/all_windows.fasta')
    ap.add_argument('--emb-dir', default='origin_data/substate/out/')
    ap.add_argument('--out', default='origin_data/joint_hetero_graph_motif_esmc.pt',
                    help='输出图（带 esmc_x，训练默认加载此文件）')
    ap.add_argument('--force', action='store_true', help='输入图已有 esmc_x 时覆盖重写')
    args = ap.parse_args()

    if not os.path.isdir(args.emb_dir):
        logging.error(f"嵌入目录不存在: {args.emb_dir}（请先把 substate/out 拷到项目里）")
        return

    data = torch.load(args.graph, map_location='cpu', weights_only=False)
    site_names = list(data['site'].node_names)
    window = data['site'].window
    num_site = len(site_names)
    if getattr(data['site'], 'esmc_x', None) is not None and not args.force:
        logging.info("输入图已有 esmc_x，跳过（--force 覆盖重写）")
        return

    # 1. 读 all_windows.fasta: header -> 序列；序列 -> 首个 header（同序列代表位点）
    name_by_seq = {}
    with open(args.fasta) as f:
        name, chunks = None, []

        def flush():
            if name is not None:
                seq = norm_seq(''.join(chunks))
                name_by_seq.setdefault(seq, name)

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                flush()
                name = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line)
        flush()
    logging.info(f"all_windows.fasta: {len(name_by_seq)} 条唯一窗口序列")

    # 2. 读 .npy 嵌入（各层平均 → [d]；不依赖具体层数，兼容 34/35/36 任意子集）
    emb_cache = {}
    d_model = None
    for fn in sorted(os.listdir(args.emb_dir)):
        if not fn.endswith('.npy'):
            continue
        name = fn[:-4]
        try:
            arr = np.load(os.path.join(args.emb_dir, fn), allow_pickle=True).item()
        except Exception as e:
            logging.warning(f"读取 {fn} 失败，跳过: {e}")
            continue
        layers = arr.get('mean', {}) if isinstance(arr, dict) else {}
        if not layers:
            continue
        vec = np.mean([np.asarray(v, dtype=np.float32) for v in layers.values()], axis=0)
        emb_cache[name] = vec
        if d_model is None:
            d_model = vec.shape[0]
        elif vec.shape[0] != d_model:
            logging.warning(f"{fn} 维度 {vec.shape[0]} ≠ {d_model}，跳过")
            del emb_cache[name]
    if not emb_cache:
        logging.error("未读到任何嵌入，请检查 --emb-dir 指向 esmc_extractor.py 的输出目录")
        return
    logging.info(f"ESMC 嵌入: {len(emb_cache)} 个窗口，维度 {d_model}")

    # 3. 逐位点匹配（a. 位点名直配 → b. 窗口序列代表匹配）
    esmc = np.zeros((num_site, d_model), dtype=np.float32)
    matched = np.zeros(num_site, dtype=bool)
    n_name = n_seq = 0
    for i, sname in enumerate(site_names):
        vec = emb_cache.get(sname)
        if vec is not None:
            esmc[i] = vec
            matched[i] = True
            n_name += 1
            continue
        rep = name_by_seq.get(window_to_seq(window[i]))
        if rep is not None and rep in emb_cache:
            esmc[i] = emb_cache[rep]
            matched[i] = True
            n_seq += 1
    missed = int((~matched).sum())
    logging.info(f"匹配结果: 位点名直配 {n_name}，序列代表匹配 {n_seq}，缺失 {missed} "
                 f"({missed / num_site:.1%})")
    if missed > 0.05 * num_site:
        logging.warning("缺失超过 5%——请检查 all_windows.fasta 与图的位点覆盖/上游 Total.fasta 是否一致")

    # 4. 缺失位点用已匹配位点的均值填充；全表 per-dim z-score 标准化
    if missed:
        esmc[~matched] = esmc[matched].mean(axis=0)
    mu = esmc.mean(axis=0)
    sd = esmc.std(axis=0) + 1e-6
    esmc = (esmc - mu) / sd

    data['site'].esmc_x = torch.from_numpy(esmc)
    torch.save(data, args.out)
    logging.info(f"已写入 data['site'].esmc_x [{num_site}, {d_model}]，保存为 {args.out}"
                 f"（训练/评估脚本默认加载此图）")


if __name__ == '__main__':
    main()
