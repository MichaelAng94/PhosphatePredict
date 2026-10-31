#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-kinase 失败模式分析：固定评估协议 + 全位点排序 + 家族/度数分层。

用法（项目根目录）：
    python3 train/analyze_per_kinase.py model/best_joint_model_new.pt
    python3 train/analyze_per_kinase.py model/best_joint_model_new.pt --split both

默认只分析 test；val 始终仅用于校准 F1 阈值。输出：
    per_kinase.csv       每个激酶的 PR-AUC/AUC/F1、固定候选排序、全位点 Recall@K
    family_summary.csv   家族宏平均与按正边数加权指标
    degree_summary.csv   按已知底物度数、PPI 度数、相似边度数分层
    failure_report.md    最差激酶/家族、度数趋势与诊断结论

说明：全位点排序把除已知正边外的位点视为未标注负例，Recall@K 可能受未知真边影响而偏保守；
该指标用于相对比较激酶/家族，不等同于实验验证精度。分析不修改 checkpoint 或图数据。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_checkpoint import build_protocol
from train_hgt_trans import (
    DEVICE, KINASE_FAMILY_FILE, PROJECT_ROOT, load_kinase_family_labels,
    select_best_f1_threshold,
)
from model.hgt_transformer import HGTTransformer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', nargs='?', default=str(PROJECT_ROOT / 'model/best_joint_model_new.pt'))
    p.add_argument('--split', choices=('test', 'val', 'both'), default='test')
    p.add_argument('--outdir', default=str(PROJECT_ROOT / 'analysis/per_kinase'))
    p.add_argument('--rank-batch-size', type=int, default=8192)
    p.add_argument('--no-full-ranking', action='store_true', help='跳过全位点排序，只分析固定评估样本')
    return p.parse_args()


def load_family_names(kinase_names):
    """返回与图节点对齐的家族名字符串；无标签记为 UNLABELED。"""
    df = pd.read_excel(KINASE_FAMILY_FILE)
    mapping = {
        str(row['Name']).strip().upper(): str(row['Family']).strip()
        for _, row in df.iterrows()
    }
    return np.array([mapping.get(str(k).strip().upper(), 'UNLABELED') for k in kinase_names])


def predict_fixed(model, k_emb, s_emb, pairs):
    """固定样本预测；保持 pos/neg 原始顺序，供按激酶与按正边分组。"""
    pos_k, pos_s, neg_k, neg_s = pairs
    with torch.no_grad():
        pos_score = model.predict_interaction(k_emb, s_emb, pos_k, pos_s).cpu().numpy()
        neg_score = model.predict_interaction(k_emb, s_emb, neg_k, neg_s).cpu().numpy()
    return {
        'pos_k': pos_k.cpu().numpy(), 'pos_s': pos_s.cpu().numpy(),
        'neg_k': neg_k.cpu().numpy(), 'neg_s': neg_s.cpu().numpy(),
        'pos_score': pos_score, 'neg_score': neg_score,
    }


def kinase_degrees(data, num_kinase):
    """PPI/ESM 相似边度数；这些边在 train_data 中未被 cold-start 删除。"""
    result = {}
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        if src != 'kinase' or dst != 'kinase':
            continue
        if 'similar' in rel.lower():
            name = 'similar_degree'
        elif 'ppi' in rel.lower() or 'interact' in rel.lower():
            name = 'ppi_degree'
        else:
            continue
        ei = data[edge_type].edge_index.cpu().numpy()
        result[name] = np.bincount(ei[0], minlength=num_kinase)
    for name in ('ppi_degree', 'similar_degree'):
        result.setdefault(name, np.zeros(num_kinase, dtype=np.int64))
    return result


def full_site_ranking(model, k_emb, s_emb, kinase_id, positive_sites, batch_size):
    """一个激酶对全部位点排序，返回 recall@K、MRR、中位/平均正例排名。"""
    num_site = s_emb.size(0)
    scores = np.empty(num_site, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, num_site, batch_size):
            end = min(start + batch_size, num_site)
            ss = torch.arange(start, end, device=s_emb.device)
            kk = torch.full_like(ss, int(kinase_id))
            scores[start:end] = model.predict_interaction(k_emb, s_emb, kk, ss).cpu().numpy()
    positive_sites = np.unique(positive_sites)
    pos_scores = scores[positive_sites]
    # 并列时按乐观排名；仅用于相对比较，不改变整体结论
    ranks = 1 + np.sum(scores[None, :] > pos_scores[:, None], axis=1)
    return {
        'full_recall@20': float(np.mean(ranks <= 20)),
        'full_recall@50': float(np.mean(ranks <= 50)),
        'full_recall@100': float(np.mean(ranks <= 100)),
        'full_mrr': float(np.mean(1.0 / ranks)),
        'full_median_rank': float(np.median(ranks)),
        'full_mean_rank': float(np.mean(ranks)),
        'full_mean_pos_score': float(np.mean(pos_scores)),
        'full_mean_other_score': float(np.mean(np.delete(scores, positive_sites))),
    }


def per_kinase_metrics(model, k_emb, s_emb, fixed, kinase_names, family_names,
                       degrees, threshold, full_ranking, rank_batch_size):
    rows = []
    pos_k, neg_k = fixed['pos_k'], fixed['neg_k']
    n_pos = len(pos_k)
    n_neg_rounds = len(neg_k) // n_pos
    # build_fixed_eval_pairs 是逐轮拼接负样本：第 r 轮对应 neg[r*n_pos:(r+1)*n_pos]
    neg_by_pos = fixed['neg_score'].reshape(n_neg_rounds, n_pos).T

    for kid in np.unique(pos_k):
        pos_mask = pos_k == kid
        neg_mask = neg_k == kid
        pred = np.concatenate([fixed['pos_score'][pos_mask], fixed['neg_score'][neg_mask]])
        label = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
        candidate_ranks = 1 + np.sum(neg_by_pos[pos_mask] > fixed['pos_score'][pos_mask, None], axis=1)
        row = {
            'kinase_id': int(kid),
            'kinase': kinase_names[kid],
            'family': family_names[kid],
            'n_pos': int(pos_mask.sum()),
            'n_neg': int(neg_mask.sum()),
            'ppi_degree': int(degrees['ppi_degree'][kid]),
            'similar_degree': int(degrees['similar_degree'][kid]),
            'pr_auc': average_precision_score(label, pred),
            'auc': roc_auc_score(label, pred),
            'f1': float(np.mean((pred >= threshold) == label)),  # 下方覆盖为真正 F1
            'precision': float(np.sum(label[pred >= threshold] == 1) / max(np.sum(pred >= threshold), 1)),
            'recall': float(np.mean(pred[label == 1] >= threshold)),
            'candidate_hits@1': float(np.mean(candidate_ranks == 1)),
            'candidate_mrr': float(np.mean(1.0 / candidate_ranks)),
            'candidate_mean_rank': float(np.mean(candidate_ranks)),
            'mean_pos_score': float(np.mean(fixed['pos_score'][pos_mask])),
            'mean_neg_score': float(np.mean(fixed['neg_score'][neg_mask])),
        }
        tp = int(np.sum((pred >= threshold) & (label == 1)))
        fp = int(np.sum((pred >= threshold) & (label == 0)))
        fn = int(np.sum((pred < threshold) & (label == 1)))
        row['f1'] = 2 * tp / max(2 * tp + fp + fn, 1)
        if full_ranking:
            row.update(full_site_ranking(
                model, k_emb, s_emb, kid, fixed['pos_s'][pos_mask], rank_batch_size
            ))
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_mean(df, col, weight='n_pos'):
    valid = df[col].notna()
    if not valid.any():
        return np.nan
    return float(np.average(df.loc[valid, col], weights=df.loc[valid, weight]))


def summarize_families(df):
    metric_cols = [c for c in (
        'pr_auc', 'auc', 'f1', 'recall', 'candidate_mrr',
        'full_recall@20', 'full_recall@50', 'full_recall@100', 'full_mrr'
    ) if c in df.columns]
    rows = []
    for family, g in df.groupby('family', sort=False):
        row = {
            'family': family, 'n_kinases': len(g), 'n_pos': int(g['n_pos'].sum()),
            'mean_n_pos': g['n_pos'].mean(), 'mean_ppi_degree': g['ppi_degree'].mean(),
        }
        for col in metric_cols:
            row[f'{col}_macro'] = g[col].mean()
            row[f'{col}_weighted'] = weighted_mean(g, col)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('pr_auc_weighted').reset_index(drop=True)


def degree_bin(series, kind):
    if kind == 'substrate':
        bins, labels = [-1, 5, 20, 50, np.inf], ['1-5', '6-20', '21-50', '>50']
    else:
        bins, labels = [-1, 0, 2, 5, 10, np.inf], ['0', '1-2', '3-5', '6-10', '>10']
    return pd.cut(series, bins=bins, labels=labels)


def summarize_degrees(df):
    frames = []
    metrics = [c for c in ('pr_auc', 'auc', 'f1', 'recall', 'full_recall@20',
                           'full_recall@50', 'full_recall@100', 'full_mrr') if c in df.columns]
    for col, kind, name in (
        ('n_pos', 'substrate', 'known_substrates'),
        ('ppi_degree', 'other', 'PPI_degree'),
        ('similar_degree', 'other', 'similar_degree'),
    ):
        tmp = df.copy()
        tmp['degree_bin'] = degree_bin(tmp[col], kind)
        for b, g in tmp.groupby('degree_bin', observed=False):
            if g.empty:
                continue
            row = {'degree_type': name, 'degree_bin': str(b),
                   'n_kinases': len(g), 'n_pos': int(g['n_pos'].sum())}
            for metric in metrics:
                row[f'{metric}_macro'] = g[metric].mean()
                row[f'{metric}_weighted'] = weighted_mean(g, metric)
            frames.append(row)
    return pd.DataFrame(frames)


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or len(np.unique(x[valid])) < 2 or len(np.unique(y[valid])) < 2:
        return np.nan
    xr = pd.Series(x[valid]).rank().to_numpy()
    yr = pd.Series(y[valid]).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def markdown_view(df, columns):
    """精简报告表格；无 tabulate 时退化为纯文本表，不影响 CSV 输出。"""
    cols = [c for c in columns if c in df.columns]
    view = df[cols].copy()
    for col in view.select_dtypes(include='number').columns:
        view[col] = view[col].map(lambda x: f'{x:.4f}' if isinstance(x, (float, np.floating)) else x)
    try:
        return view.to_markdown(index=False)
    except ImportError:
        return '```\n' + view.to_string(index=False) + '\n```'


def write_report(df, fam, deg, split, threshold, outdir):
    overall_pr = weighted_mean(df, 'pr_auc')
    lines = [
        f'# Per-kinase 失败模式分析（{split}）', '',
        f'- 激酶数：{len(df)}；正边数：{int(df.n_pos.sum())}；val 校准阈值：{threshold:.4f}',
        f'- 按正边数加权 PR-AUC：{overall_pr:.4f}；激酶宏平均 PR-AUC：{df.pr_auc.mean():.4f}',
        '- PR-AUC 基于每个正边配 5 个随机 tail 负样本，随机基线约为 1/6；跨激酶比较时需注意各自候选难度。',
        '- full_recall@K 为该激酶在全部位点中的排序召回；未标注位点视为负例，因此数值偏保守。', '',
        '## 1. 最拖后腿的激酶（按 PR-AUC 升序，前 15）', '',
        markdown_view(df.sort_values('pr_auc').head(15), [
            'kinase', 'family', 'n_pos', 'ppi_degree', 'similar_degree',
            'pr_auc', 'auc', 'f1', 'recall', 'candidate_mrr',
            'full_recall@50', 'full_median_rank',
        ]), '',
        '## 2. 家族分层（按加权 PR-AUC 升序）', '',
        '小样本家族的指标波动很大；优先看 `n_kinases>=2` 且 `n_pos>=20` 的家族，勿对单激酶家族过度解读。', '',
        markdown_view(fam, [
            'family', 'n_kinases', 'n_pos', 'pr_auc_weighted', 'pr_auc_macro',
            'pr_auc_delta_vs_overall', 'pr_auc_contribution',
            'full_recall@50_weighted', 'full_mrr_weighted',
        ]), '',
        '## 3. 度数分层', '',
        markdown_view(deg, [
            'degree_type', 'degree_bin', 'n_kinases', 'n_pos',
            'pr_auc_weighted', 'pr_auc_macro', 'full_recall@50_weighted',
        ]), '',
        '## 4. 相关性诊断（Spearman）', '',
    ]
    for x in ('n_pos', 'ppi_degree', 'similar_degree'):
        lines.append(f'- {x} vs PR-AUC: {spearman(df[x], df.pr_auc):.4f}')
        if 'full_recall@50' in df:
            lines.append(f'- {x} vs full_recall@50: {spearman(df[x], df["full_recall@50"]):.4f}')
    lines += ['', '## 5. 解读规则', '',
        '- 低 PR-AUC + 低 full_recall@K：排序本身失败，优先检查该激酶的家族迁移、相似边和输入特征。',
        '- PR-AUC 尚可 + F1/Recall 很低：主要是分数校准/阈值问题，不宜直接判为表征失败。',
        '- `pr_auc_delta_vs_overall<0` 的家族低于整体；`pr_auc_contribution` 越负，对整体加权 PR-AUC 的拖累越大。',
        '- 差激酶集中于同一家族：提示家族原型/家族先验不足，而不是单纯低度数问题。',
        '- 差激酶集中于低 PPI/相似边度数：提示结构上下文不足；可重点检查这些节点的边覆盖。',
        '- 仅 `n_pos` 很小的激酶差：小样本方差和底物数量稀疏共同作用，需结合多种子结果判断。', '']
    (outdir / 'failure_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = ('val', 'test') if args.split == 'both' else (args.split,)

    train_data, fixed_pairs, dims, family_group_map = build_protocol()
    esm_dim, n2v_dim, site_dim, num_families, motif_len, num_groups = dims
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    hidden_dim = int(state['family_prototypes'].shape[1])
    model = HGTTransformer(
        esm_dim=esm_dim, n2v_dim=n2v_dim, site_dim=site_dim,
        num_families=num_families, hidden_dim=hidden_dim, num_heads=4,
        num_hgt_layers=2, num_transformer_layers=1, dropout=0.2,
        motif_len=motif_len, num_groups=num_groups, family_group_map=family_group_map,
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    kinase_names = train_data['kinase'].node_names
    family_names = load_family_names(kinase_names)
    degrees = kinase_degrees(train_data, len(kinase_names))
    with torch.no_grad():
        emb = model(train_data)
        k_emb, s_emb = emb['kinase'], emb['site']

    fixed_preds = {s: predict_fixed(model, k_emb, s_emb, fixed_pairs[s])
                   for s in set(splits) | {'val'}}
    vp = fixed_preds['val']
    threshold, _ = select_best_f1_threshold(
        np.concatenate([np.ones(len(vp['pos_score'])), np.zeros(len(vp['neg_score']))]),
        np.concatenate([vp['pos_score'], vp['neg_score']]),
    )
    print(f'val 校准阈值: {threshold:.4f}')

    for split in splits:
        print(f'分析 {split}，逐激酶指标{" + 全位点排序" if not args.no_full_ranking else ""}...')
        df = per_kinase_metrics(
            model, k_emb, s_emb, fixed_preds[split], kinase_names, family_names,
            degrees, threshold, not args.no_full_ranking, args.rank_batch_size,
        )
        df = df.sort_values('pr_auc').reset_index(drop=True)
        fam = summarize_families(df)
        overall_pr = weighted_mean(df, 'pr_auc')
        # delta<0 表示低于整体；contribution 是该家族对整体加权 PR-AUC 的相对贡献，越负越拖后腿
        fam['pr_auc_delta_vs_overall'] = fam['pr_auc_weighted'] - overall_pr
        fam['pr_auc_contribution'] = (fam['n_pos'] / df['n_pos'].sum()) * fam['pr_auc_delta_vs_overall']
        deg = summarize_degrees(df)
        suffix = '' if len(splits) == 1 else f'_{split}'
        df.to_csv(outdir / f'per_kinase{suffix}.csv', index=False, encoding='utf-8-sig')
        fam.to_csv(outdir / f'family_summary{suffix}.csv', index=False, encoding='utf-8-sig')
        deg.to_csv(outdir / f'degree_summary{suffix}.csv', index=False, encoding='utf-8-sig')
        write_report(df, fam, deg, split, threshold, outdir)
        if len(splits) > 1:
            (outdir / 'failure_report.md').rename(outdir / f'failure_report_{split}.md')
        print(f'{split}: 加权 PR-AUC={weighted_mean(df, "pr_auc"):.4f}, '
              f'宏平均 PR-AUC={df.pr_auc.mean():.4f}')

    print(f'结果目录: {outdir}')


if __name__ == '__main__':
    main()
