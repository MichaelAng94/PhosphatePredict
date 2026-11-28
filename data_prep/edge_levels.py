#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""家族层面边 / 自磷酸化边在训练与评估中的处理——建图脚本写入图，训练脚本与评估工具共用。

图上的标注（graph/build_joint_graph.py 写入；旧图没有这些属性，此时一切行为与原来逐边一致）：
  data['kinase', 'phosphorylates', 'site'].edge_level  LongTensor[E]
      0 = 特定激酶-位点注释      1 = 家族层面注释（激酶端是 FAMILY_xxx 节点）      2 = 自磷酸化
  data['kinase'].is_family        BoolTensor[N]         是否家族节点
  data['kinase'].family_members   list[list[int]]       家族节点 idx -> 成员节点 idx（其余为空表）

约定：
  * cold-start 划分只针对 level-0 边——验证/测试激酶在训练图里没有任何磷酸化边。
    家族节点的边、训练激酶的自磷酸化边只进训练图。评估激酶的自磷酸化边两边都不进：
    模型没有"位点在激酶自己身上"的特征，放进评估只会拉低指标。
  * "某个 PKA 磷酸化位点 s"这类家族层面注释，对家族成员既不是正例也不是负例：
    负采样与批内对比损失都不把 s 当作成员的负样本；全候选排序评估把 s 移出成员的候选。
    反过来同样成立：成员的特异注释位点不作为家族节点的负样本。
  * 检索谱的来源只取有特定激酶训练边的激酶（split['train_specific']），不含家族节点：
    家族节点的底物只有家族层面位点，而这些位点对成员既不计正也不计负，
    作为检索源只会把 softmax 权重从真正有用的近亲激酶上分走。
    相似边（similar_to）不受此限：家族节点的表示只由训练边计算，是合法的近亲来源。

训练脚本 main() 里的用法（按顺序）:
    edge_level, family_members = read_edge_levels(data)
    split = split_edges(full_edge_index, edge_level, num_kinase, RANDOM_SEED, SPLIT_MODE)
    kinase_sites, site_kinases = build_true_edge_maps(full_edge_index)
    family_keys, mask_sites = apply_family_context(kinase_sites, site_kinases, family_members, num_site)
    true_edge_keys = np.unique(np.concatenate([<全量真边键>, family_keys]))   # family_keys 为空时省略
    retrieval_kmask[split['train_specific'][0]] = True                        # 检索谱来源不含家族节点
    apply_train_edges(train_data, train_edge_index, split['train_level'])
    evaluate_full_candidate_ranking(..., mask_sites=mask_sites)     # 函数内部用 masked_sites / mask_family_sites
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

LEVEL_SPECIFIC, LEVEL_FAMILY, LEVEL_AUTOPHOS = 0, 1, 2
PHOS_KEY = ('kinase', 'phosphorylates', 'site')
REV_KEY = ('site', 'rev_phosphorylates', 'kinase')


def read_edge_levels(data):
    """返回 (edge_level, family_members)。旧图（没有 edge_level 属性）返回 (None, None)。"""
    level = getattr(data['kinase', 'phosphorylates', 'site'], 'edge_level', None)
    members = getattr(data['kinase'], 'family_members', None) if level is not None else None
    return level, members


def split_edges(full_edge_index, edge_level, num_kinase, random_seed, mode='kinase'):
    """链接预测的数据划分，只针对 level-0（特定激酶）的边。返回 dict：
        train / val / test        边索引（train 已并入全部家族层面边、非评估激酶的自磷酸化边）
        train_specific            train 中 level-0 的部分（检索谱的来源激酶取自这里）
        train_level               train 各条边的层级，与 train 同序（旧图为 None）
        train_k / val_k / test_k  划分到的激酶索引
        eval_kinase               BoolTensor[num_kinase]，验证 / 测试激酶
        n_family / n_auto         train 里并入的家族层面边 / 自磷酸化边条数
    edge_level 为 None（旧图）时与原划分逐边一致。"""
    device = full_edge_index.device
    has_level = edge_level is not None
    if not has_level:
        edge_level = torch.zeros(full_edge_index.size(1), dtype=torch.long, device=device)
    is_spec = edge_level == 0
    specific, extra = full_edge_index[:, is_spec], full_edge_index[:, ~is_spec]
    extra_level = edge_level[~is_spec]
    eval_kinase = torch.zeros(num_kinase, dtype=torch.bool, device=device)

    if mode == 'kinase':
        edge_kinases = specific[0].cpu().numpy()
        train_k, temp_k = train_test_split(np.unique(edge_kinases), test_size=0.3, random_state=random_seed)
        val_k, test_k = train_test_split(temp_k, test_size=1/3, random_state=random_seed)

        def take(names):                          # 每个激酶的全部边进入同一子集
            return specific[:, torch.as_tensor(np.isin(edge_kinases, names), device=device)]

        train_specific, val, test = take(train_k), take(val_k), take(test_k)
        eval_kinase[torch.as_tensor(np.concatenate([val_k, test_k]), device=device)] = True
        # 家族节点不在评估激酶里，其边全部保留；评估激酶的自磷酸化边不进训练图
        keep = (extra_level != LEVEL_AUTOPHOS) | ~eval_kinase[extra[0]]
    else:
        idx = np.arange(specific.size(1))
        tr, tmp = train_test_split(idx, test_size=0.3, random_state=random_seed)
        va, te = train_test_split(tmp, test_size=1/3, random_state=random_seed)
        train_specific, val, test = specific[:, tr], specific[:, va], specific[:, te]
        ek = specific[0].cpu().numpy()
        train_k, val_k, test_k = np.unique(ek[tr]), np.unique(ek[va]), np.unique(ek[te])
        eval_kinase[torch.as_tensor(np.concatenate([val_k, test_k]), device=device)] = True
        keep = torch.ones(extra.size(1), dtype=torch.bool, device=device)

    zero = torch.zeros(train_specific.size(1), dtype=torch.long, device=device)
    if extra.size(1):
        train = torch.cat([train_specific, extra[:, keep]], dim=1)
        train_level = torch.cat([zero, extra_level[keep]])
    else:
        train = train_specific
        train_level = zero if has_level else None   # 旧图不写入层级标注
    return dict(train=train, val=val, test=test, train_specific=train_specific, train_level=train_level,
                train_k=train_k, val_k=val_k, test_k=test_k, eval_kinase=eval_kinase,
                n_family=int((extra_level == LEVEL_FAMILY).sum()),
                n_auto=int((extra_level[keep] == LEVEL_AUTOPHOS).sum()))


def apply_train_edges(data, train_edge_index, train_level=None):
    """把 data 的磷酸化边（正向 + 反向，以及与之同序的 edge_level）替换成训练边，得到训练图快照。
    clone 之后必须走这里而不是只改 edge_index，否则 edge_level 仍是全量长度、与边对不上。
    train_level 传 split['train_level']；None 表示图没有层级标注（旧图），此时只换边、不动标注。"""
    data[PHOS_KEY].edge_index = train_edge_index
    data[REV_KEY].edge_index = train_edge_index[[1, 0]]
    if train_level is not None:
        level = train_level.to(train_edge_index.device)
        data[PHOS_KEY].edge_level = level
        data[REV_KEY].edge_level = level


def apply_family_context(kinase_sites, site_kinases, family_members, num_site):
    """就地补全负样本过滤索引，并给出评估时要移出候选的位点。返回 (extra_keys, mask_sites)：
      kinase_sites[k] / site_kinases[s]：build_true_edge_maps 给出的 {激酶: 位点集合} / {位点: 激酶集合}；
          补全后——
            家族层面位点的激酶集合并入全部成员，成员的位点集合并入所属家族的家族层面位点；
            家族节点的位点集合并入全部成员的位点，成员位点的激酶集合并入所属家族节点。
      extra_keys：(成员或家族节点, 位点) 的键 k*num_site+s，即上面新补的"不算负样本"的对。
          并入 true_edge_keys 后，批内对比损失与检索谱反污染也不把它们当负样本。
      mask_sites：按激酶 idx 的列表，该激酶所属家族的家族层面注释位点（numpy 数组；没有则为 None）。
    family_members 为 None（旧图）时什么都不改，返回空键与空列表。"""
    size = len(family_members) if family_members is not None else 0
    mask_sites = [None] * size
    keys = set()
    orig = {k: set(v) for k, v in kinase_sites.items()}   # 补全之前的位点集合：结果不随家族的处理顺序变化
    for f, members in enumerate(family_members or []):
        if not members:
            continue
        f_sites = orig.get(f, set())                      # 家族层面的注释位点
        own = set()
        for m in members:
            own |= orig.get(m, set())                     # 成员自己的特异注释位点
        kinase_sites[f] = f_sites | own
        keys.update(f * num_site + s for s in own)
        for s in own:                                     # 成员的特异位点：家族节点也不以其为负样本
            site_kinases[s] = site_kinases.get(s, set()) | {f}
        if f_sites:
            site_members = set(members)
            for m in members:
                kinase_sites[m] = kinase_sites.get(m, set()) | f_sites
                keys.update(m * num_site + s for s in f_sites)
                mask_sites[m] = f_sites if mask_sites[m] is None else mask_sites[m] | f_sites
            for s in f_sites:
                site_kinases[s] = site_kinases.get(s, set()) | site_members
    mask_sites = [np.array(sorted(x), dtype=np.int64) if x else None for x in mask_sites]
    return np.array(sorted(keys), dtype=np.int64), mask_sites


def masked_sites(mask_sites, k_id):
    """激酶 k_id 在全候选排序时要移出的位点（numpy 数组或 None）。
    旧图没有家族层面注释时 mask_sites 是空列表，这里返回 None 而不是越界。"""
    if mask_sites is None or k_id >= len(mask_sites):
        return None
    return mask_sites[k_id]


def mask_family_sites(scores, pos_idx, masked):
    """全候选排序里把家族层面注释位点移出某个激酶的候选：分数置 -inf（排到最后）。
    该激酶自己的特异正例不屏蔽。返回被屏蔽的位点数。
    scores: torch 张量 [num_site]；pos_idx: 该激酶正例的位点 idx；masked: 位点 idx 数组或 None。"""
    if masked is None or len(masked) == 0:
        return 0
    keep_out = np.setdiff1d(np.asarray(masked, dtype=np.int64), np.asarray(pos_idx, dtype=np.int64))
    if len(keep_out) == 0:
        return 0
    scores[torch.as_tensor(keep_out, device=scores.device)] = float("-inf")
    return int(len(keep_out))
