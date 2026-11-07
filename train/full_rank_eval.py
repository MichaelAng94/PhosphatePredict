#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全候选排序评估（共享模块）——把每个激酶放到**全部位点**上排序。

真实应用口径：给定一个（冷启动）激酶，在蛋白质组全部候选位点中排出其底物。
与随机 1:5 负采样的 PR-AUC 相比，它给出三类后者看不到的信息：
  1. 候选空间可发现性：Recall@K / Precision@K / Hits@K / MRR / 正例中位秩；
  2. 逐激酶 full AP（未标注位点按负例，保守口径），区分宏平均与按正边数加权；
  3. 按已知底物数与家族分层的失败结构。

注意：本口径把"未标注位点"全部当负例，绝对值偏低，只用于相对比较和分层诊断，
不能与随机负采样 PR-AUC 直接对比；正式主指标仍以 val 选择、test 报告为准。
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score

DEFAULT_RANK_KS = (10, 20, 50, 100)   # Recall@K 的 K 列表
DEFAULT_HIT_KS = (1, 10, 20)          # Hits@K：top-K 内至少命中一个真实底物的激酶比例
DEFAULT_SITE_CHUNK = 8192              # 位点分块大小（控制显存）
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis", "full_rank",
)


def score_kinase_all_sites(model, kinase_emb, site_emb, k_id, num_site,
                           fam_probs=None, site_chunk=DEFAULT_SITE_CHUNK):
    """单个激酶在全部位点上的打分（logit）。logit 与概率严格单调等价，排序指标不变。"""
    device = kinase_emb.device
    out = torch.empty(num_site, dtype=torch.float32, device=device)
    for start in range(0, num_site, site_chunk):
        end = min(start + site_chunk, num_site)
        s_idx = torch.arange(start, end, dtype=torch.long, device=device)
        k_idx = torch.full((end - start,), int(k_id), dtype=torch.long, device=device)
        out[start:end] = model.interaction_logits(
            kinase_emb, site_emb, k_idx, s_idx, fam_probs=fam_probs
        )
    return out


def _npos_bin(n):
    """按已知底物数分层：区分"底物丰富"与"底物稀疏"激酶的可发现性差异。"""
    n = int(n)
    if n <= 5:
        return '1-5'
    if n <= 20:
        return '6-20'
    if n <= 50:
        return '21-50'
    return '>50'


def evaluate_full_candidate_ranking(model, embed_data, edge_index, num_site,
                                    ks=DEFAULT_RANK_KS, hit_ks=DEFAULT_HIT_KS,
                                    kinase_labels=None, family_names=None,
                                    kinase_names=None, max_kinases=0,
                                    site_chunk=DEFAULT_SITE_CHUNK,
                                    split_name='test', save_csv=False,
                                    out_dir=DEFAULT_OUT_DIR, date_tag=None,
                                    verbose=True):
    """返回：标量汇总指标 dict（逐激酶明细见 CSV）。参数同 no_transformer 脚本内版本。"""
    import logging

    model.eval()
    ks = tuple(sorted({int(k) for k in ks if int(k) > 0})) or (50,)
    hit_ks = tuple(sorted({int(k) for k in hit_ks if int(k) > 0}))
    tag = date_tag or time.strftime("%Y%m%d_%H%M%S")

    with torch.no_grad():
        emb = model(embed_data)
        kinase_emb, site_emb = emb['kinase'], emb['site']
        num_site = min(int(num_site), int(site_emb.size(0)))
        # 家族概率只算一次，避免每个位点分块重复前向
        fam_probs = torch.softmax(model.predict_family(kinase_emb), dim=-1)

        ek = edge_index[0].detach().cpu().numpy()
        es = edge_index[1].detach().cpu().numpy()
        pos_by_k = defaultdict(set)
        for kv, sv in zip(ek, es):
            if sv < num_site:
                pos_by_k[int(kv)].add(int(sv))

        items = sorted(pos_by_k.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        n_kin_total = len(items)
        if max_kinases and int(max_kinases) > 0:
            items = items[:int(max_kinases)]

        rows = []
        pooled_scores, pooled_labels, pooled_ranks = [], [], []
        for k_id, site_set in tqdm(items, desc=f'全候选排序[{split_name}]', leave=False):
            scores = score_kinase_all_sites(
                model, kinase_emb, site_emb, k_id, num_site,
                fam_probs=fam_probs, site_chunk=site_chunk
            ).cpu().numpy()
            # 一次性排出全部位点名次（1 为最高分）；并列按稳定排序处理
            order = np.argsort(-scores, kind='stable')
            rank = np.empty(num_site, dtype=np.int64)
            rank[order] = np.arange(1, num_site + 1)

            pos = np.fromiter(site_set, dtype=np.int64, count=len(site_set))
            pos_ranks = rank[pos]
            n_pos = int(pos.size)

            label_vec = np.zeros(num_site, dtype=np.int8)
            label_vec[pos] = 1
            ap = float(average_precision_score(label_vec, scores))

            fam_id = -1
            if kinase_labels is not None:
                fam_id = int(kinase_labels[k_id].item()) if torch.is_tensor(kinase_labels) \
                    else int(kinase_labels[k_id])
            fam_name = (family_names[fam_id] if (family_names is not None and fam_id >= 0)
                        else 'UNLABELED')
            k_name = (str(kinase_names[k_id]) if kinase_names is not None else str(k_id))

            row = {
                'kinase_id': int(k_id), 'kinase': k_name, 'family': fam_name,
                'family_id': fam_id, 'n_pos': n_pos, 'n_site': num_site,
                'full_ap': ap,
                'mrr': float(np.mean(1.0 / pos_ranks)),
                'median_rank': float(np.median(pos_ranks)),
                'mean_rank': float(np.mean(pos_ranks)),
                'best_rank': int(pos_ranks.min()),
            }
            for K in ks:
                hit_cnt = int((pos_ranks <= K).sum())
                row[f'recall@{K}'] = hit_cnt / n_pos
                row[f'precision@{K}'] = hit_cnt / K
            for K in hit_ks:
                row[f'hit@{K}'] = float(bool((pos_ranks <= K).any()))
            rows.append(row)

            pooled_scores.append(scores.astype(np.float32, copy=False))
            pooled_labels.append(label_vec)
            pooled_ranks.append(pos_ranks)

        del fam_probs

    if not rows:
        logging.warning(f"[全候选/{split_name}] 该 split 无正边，跳过")
        return {}

    df = pd.DataFrame(rows)
    n_pos_total = int(df['n_pos'].sum())
    w = df['n_pos'].to_numpy(dtype=np.float64)
    w_sum = float(w.sum())

    def _macro(col):
        return float(df[col].mean())

    def _wmean(col):
        return float((df[col].to_numpy(dtype=np.float64) * w).sum() / w_sum)

    pooled_ap = float(average_precision_score(
        np.concatenate(pooled_labels), np.concatenate(pooled_scores)
    ))
    # 正例层面的池化统计（每条真边等权）：比"逐激酶中位秩再加权平均"更易解释
    all_pos_ranks = np.concatenate(pooled_ranks).astype(np.float64)
    pooled_median_rank = float(np.median(all_pos_ranks))
    pooled_mrr = float(np.mean(1.0 / all_pos_ranks))
    del pooled_labels, pooled_scores, pooled_ranks, all_pos_ranks

    fr = {
        'n_kinases': int(len(df)), 'n_kinases_total': int(n_kin_total),
        'n_pos': n_pos_total, 'n_site': int(num_site),
        'macro_ap': _macro('full_ap'), 'weighted_ap': _wmean('full_ap'),
        'pooled_ap': pooled_ap,
        'macro_mrr': _macro('mrr'), 'weighted_mrr': _wmean('mrr'),
        'pooled_mrr': pooled_mrr,
        'pooled_median_rank': pooled_median_rank,
    }
    for K in ks:
        fr[f'macro_recall@{K}'] = _macro(f'recall@{K}')
        fr[f'weighted_recall@{K}'] = _wmean(f'recall@{K}')
    for K in hit_ks:
        fr[f'hit@{K}_rate'] = _macro(f'hit@{K}')

    if verbose:
        ks_str = ' / '.join(
            f"@{K} {fr[f'weighted_recall@{K}']:.3f}(宏 {fr[f'macro_recall@{K}']:.3f})"
            for K in ks
        )
        hit_str = ' / '.join(f"@{K} {fr[f'hit@{K}_rate']:.3f}" for K in hit_ks)
        logging.info(
            f"[全候选/{split_name}] 激酶 {fr['n_kinases']}/{fr['n_kinases_total']} | "
            f"正边 {n_pos_total} | 候选位点 {num_site}"
        )
        logging.info(
            f"[全候选/{split_name}] AP: 宏平均 {fr['macro_ap']:.4f} | "
            f"按正边加权 {fr['weighted_ap']:.4f} | 池化 {fr['pooled_ap']:.4f}"
        )
        logging.info(
            f"[全候选/{split_name}] Recall(加权/宏) {ks_str} | "
            f"MRR 池化 {fr['pooled_mrr']:.4f}/加权 {fr['weighted_mrr']:.4f}/宏 {fr['macro_mrr']:.4f} | "
            f"正例池化中位秩 {fr['pooled_median_rank']:.0f}"
        )
        logging.info(f"[全候选/{split_name}] Hits@K（激酶命中率）{hit_str}")

        # 分层：已知底物数
        for b in ('1-5', '6-20', '21-50', '>50'):
            sub = df[df['n_pos'].map(_npos_bin) == b]
            if sub.empty:
                continue
            sw = sub['n_pos'].to_numpy(dtype=np.float64)
            k50 = 'recall@50' if 'recall@50' in df.columns else f"recall@{ks[-1]}"
            logging.info(
                f"[全候选/{split_name}] 分层 n_pos={b}: 激酶 {len(sub)}, 正边 {int(sw.sum())}, "
                f"AP 宏 {sub['full_ap'].mean():.4f}/加权 "
                f"{(sub['full_ap'].to_numpy() * sw).sum() / max(sw.sum(), 1e-9):.4f}, "
                f"Recall@50 加权 {(sub[k50].to_numpy() * sw).sum() / max(sw.sum(), 1e-9):.4f}"
            )

        # 分层：家族（按加权 AP 升序的前 3 个——失败集中区）
        fam_rows = {}
        for fam_name, sub in df.groupby('family'):
            sw = sub['n_pos'].to_numpy(dtype=np.float64)
            fam_rows[fam_name] = {
                'n_kinases': int(len(sub)), 'n_pos': int(sw.sum()),
                'ap_weighted': float((sub['full_ap'].to_numpy() * sw).sum() / max(sw.sum(), 1e-9)),
                'ap_macro': float(sub['full_ap'].mean()),
                'recall@50_weighted': float((sub['recall@50'].to_numpy() * sw).sum() / max(sw.sum(), 1e-9))
                if 'recall@50' in df.columns else float('nan'),
                'mrr_weighted': float((sub['mrr'].to_numpy() * sw).sum() / max(sw.sum(), 1e-9)),
            }
        fam_df = pd.DataFrame(fam_rows).T.astype(float).sort_values('ap_weighted')
        worst = fam_df.head(3)
        logging.info(
            f"[全候选/{split_name}] 家族拖累 Top3（按加权 AP）: " + ", ".join(
                f"{name}={r['ap_weighted']:.3f}(n_pos={int(r['n_pos'])})"
                for name, r in worst.iterrows()
            )
        )

        if save_csv:
            os.makedirs(out_dir, exist_ok=True)
            p_kin = os.path.join(out_dir, f"{split_name}_per_kinase_{tag}.csv")
            df.sort_values('full_ap').to_csv(p_kin, index=False, encoding='utf-8-sig')
            p_fam = os.path.join(out_dir, f"{split_name}_family_{tag}.csv")
            fam_df.to_csv(p_fam, encoding='utf-8-sig')
            logging.info(f"[全候选/{split_name}] 明细已保存: {p_kin} | {p_fam}")

    return fr


def format_full_rank_short(fr, split_name='val'):
    """训练期紧凑日志：只报最能反映"真实可发现性"的几个量。"""
    if not fr:
        return '不适用'
    parts = [f"AP宏 {fr['macro_ap']:.4f}", f"AP加权 {fr['weighted_ap']:.4f}"]
    for K in (50, 20, 10):
        key = f'weighted_recall@{K}'
        if key in fr:
            parts.append(f"R@{K} {fr[key]:.3f}")
            break
    for K in (10, 20, 1):
        key = f'hit@{K}_rate'
        if key in fr:
            parts.append(f"Hits@{K} {fr[key]:.3f}")
            break
    parts.append(f"MRR {fr['weighted_mrr']:.3f}")
    return ' | '.join(parts)
