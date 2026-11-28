#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_prep/edge_levels.py 的测试（需要 torch；合成用例不依赖 torch_geometric）。

用法（WSL 项目根目录，重建图之前、之后都可以跑）:
    python3 data_prep/test_edge_levels.py [图路径，默认 origin_data/joint_hetero_graph.pt] [-v]

两组用例：
  A 合成图：划分、负样本过滤、评估屏蔽、训练图快照的各项约定，逐条断言。
  B 真实图（默认 origin_data/joint_hetero_graph.pt）：
    旧图（没有 edge_level 属性）：新的划分代码必须与原训练脚本里的划分代码逐边一致——
      "旧图上行为不变"的唯一硬证据；图是已知的老图（567 个激酶节点）时再核对划分 11601 / 2935 / 1680。
    新图（带 edge_level）：特定激酶的边与原代码逐边一致地划入 train / val / test；家族层面边与训练激酶的
      自磷酸化边只出现在训练图里；家族层面的位点不会被采样成成员的负样本。
图文件不存在时 B 自动跳过并说明原因（不算失败）。
"""

import argparse
import os
import sys
import unittest
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_prep.edge_levels import (
    PHOS_KEY, REV_KEY, apply_family_context, apply_train_edges, mask_family_sites, masked_sites,
    read_edge_levels, split_edges,
)

DEFAULT_GRAPH = os.path.join(_PROJECT_ROOT, 'origin_data', 'joint_hetero_graph.pt')
GRAPH_PATH = DEFAULT_GRAPH
OLD_GRAPH_NODES, OLD_GRAPH_SPLIT = 567, (11601, 2935, 1680)   # 已知老图的节点数与 seed=42 的划分边数
SEEDS = (42, 0, 7, 2024)

# ---------------------------------------------------------------- 合成图
NUM_SITE = 20
# 激酶 0..9 是真实激酶；10 = 家族 A（成员 0, 1, 2）；11 = 家族 B（成员 2, 3, 9；激酶 2 同时属于两个家族）
FAMILY_MEMBERS = [[] for _ in range(10)] + [[0, 1, 2], [2, 3, 9]]
SPECIFIC = [(0, 0), (0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (5, 8),
            (6, 9), (7, 10), (8, 11), (9, 12)]
FAMILY = [(10, 1), (10, 13), (11, 5), (11, 14)]      # 位点 1 既是家族 A 的家族层面位点，又是激酶 0 的特异位点
AUTOPHOS = [(0, 15), (0, 17), (5, 16)]


def build_synth(with_extras=True):
    """返回 (edge_index, edge_level)。边按固定方式交错排列，不假设同一层级的边在边表里连续。"""
    rows = [(k, s, 0) for k, s in SPECIFIC]
    if with_extras:
        rows += [(k, s, 1) for k, s in FAMILY] + [(k, s, 2) for k, s in AUTOPHOS]
    rows.sort(key=lambda r: (r[1] % 3, r[0], r[1]))
    ei = torch.tensor([r[:2] for r in rows], dtype=torch.long).t().contiguous()
    return ei, torch.tensor([r[2] for r in rows], dtype=torch.long)


def true_maps(edge_index):
    """与训练脚本的 build_true_edge_maps 相同：{激酶: 位点集合}、{位点: 激酶集合}。"""
    kinase_sites, site_kinases = defaultdict(set), defaultdict(set)
    for k, s in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        kinase_sites[k].add(s)
        site_kinases[s].add(k)
    return kinase_sites, site_kinases


def keys_of(t):
    return set(zip(t[0].tolist(), t[1].tolist()))


def corrupt(fixed_side, num_candidates, true_neighbors):
    """与训练脚本的 _corrupt_and_filter 相同：随机换一侧，命中已知邻居则重采样。"""
    rng = np.random.RandomState(0)
    cand = rng.randint(0, num_candidates, size=len(fixed_side))
    for _ in range(100):
        collide = np.fromiter((int(c) in true_neighbors[int(f)] for f, c in zip(fixed_side, cand)),
                             dtype=bool, count=len(fixed_side))
        if not collide.any():
            break
        cand[collide] = rng.randint(0, num_candidates, size=int(collide.sum()))
    return cand


def legacy_split(full_edge_index, random_seed):
    """改造前训练脚本 main() 里的划分代码，逐字保留，作为对照。"""
    device = full_edge_index.device
    edge_kinases = full_edge_index[0].cpu().numpy()
    kinases_with_edges = np.unique(edge_kinases)
    train_k, temp_k = train_test_split(kinases_with_edges, test_size=0.3, random_state=random_seed)
    val_k, test_k = train_test_split(temp_k, test_size=1/3, random_state=random_seed)
    train_edge_index = full_edge_index[:, torch.as_tensor(np.isin(edge_kinases, train_k), device=device)]
    val_edge_index = full_edge_index[:, torch.as_tensor(np.isin(edge_kinases, val_k), device=device)]
    test_edge_index = full_edge_index[:, torch.as_tensor(np.isin(edge_kinases, test_k), device=device)]
    return train_edge_index, val_edge_index, test_edge_index, train_k, val_k, test_k


class _Store(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Graph(dict):
    """够用的 HeteroData 替身（不需要 torch_geometric）。"""

    def __getitem__(self, key):
        return super().setdefault(key, _Store())


class TestSyntheticGraph(unittest.TestCase):
    """A. 合成图上的各项约定。"""

    @classmethod
    def setUpClass(cls):
        cls.ei, cls.level = build_synth()
        cls.fam_keys, cls.auto_keys = set(FAMILY), set(AUTOPHOS)
        cls.spec_keys = keys_of(cls.ei) - cls.fam_keys - cls.auto_keys
        assert len(cls.spec_keys) == len(SPECIFIC)

    def test_split_kinase_partition(self):
        sp = split_edges(self.ei, self.level, 12, 42, 'kinase')
        tr_s, va, te = keys_of(sp['train_specific']), keys_of(sp['val']), keys_of(sp['test'])
        train_k, val_k, test_k = set(sp['train_k'].tolist()), set(sp['val_k'].tolist()), set(sp['test_k'].tolist())
        eval_k = sp['eval_kinase']
        # 1) 特定激酶的边完整、不重叠地分到三份；每个激酶的全部边在同一份
        self.assertEqual(tr_s | va | te, self.spec_keys)
        self.assertFalse(tr_s & va or tr_s & te or va & te)
        self.assertFalse(train_k & val_k or train_k & test_k or val_k & test_k)
        self.assertTrue(all(k in train_k for k, _ in tr_s))
        self.assertTrue(all(k in val_k for k, _ in va) and all(k in test_k for k, _ in te))
        self.assertTrue(eval_k[list(val_k | test_k)].all() and not eval_k[list(train_k)].any())
        # 2) 评估激酶在训练图里没有任何 level-0 的边
        self.assertFalse(any(bool(eval_k[k]) for k, _ in tr_s))
        # 3) 家族节点不是评估激酶，其边全部进训练图
        self.assertFalse(bool(eval_k[10]) or bool(eval_k[11]))
        self.assertTrue(self.fam_keys <= keys_of(sp['train']))
        # 4) 评估激酶的自磷酸化边两边都不进；其余自磷酸化边进训练图
        kept_auto = {p for p in self.auto_keys if not eval_k[p[0]]}
        self.assertEqual(keys_of(sp['train']), tr_s | self.fam_keys | kept_auto)
        self.assertFalse((self.auto_keys - kept_auto) & (keys_of(sp['train']) | va | te))
        self.assertEqual((sp['n_family'], sp['n_auto']), (len(self.fam_keys), len(kept_auto)))
        # 5) train_level 与 train 同序，逐条给出正确的层级
        tl = sp['train_level']
        self.assertIsNotNone(tl)
        self.assertEqual(tl.numel(), sp['train'].size(1))
        for j, (k, s) in enumerate(zip(sp['train'][0].tolist(), sp['train'][1].tolist())):
            want = 0 if (k, s) in tr_s else (1 if (k, s) in self.fam_keys else 2)
            self.assertEqual(int(tl[j]), want, f'第 {j} 条训练边的层级')
            if want == 2:
                self.assertIn((k, s), kept_auto)
        # 6) train = train_specific 在前，并入的边在后
        n_spec = sp['train_specific'].size(1)
        self.assertTrue(torch.equal(sp['train'][:, :n_spec], sp['train_specific']))
        self.assertTrue(torch.equal(tl[:n_spec], torch.zeros(n_spec, dtype=torch.long)))
        # 7) 结果确定
        sp2 = split_edges(self.ei, self.level, 12, 42, 'kinase')
        self.assertTrue(torch.equal(sp['train'], sp2['train']) and torch.equal(sp['val'], sp2['val'])
                        and torch.equal(sp['test'], sp2['test']))

    def test_split_matches_legacy_on_level0_graph(self):
        """只有特定激酶的边时，新代码与原划分逐边一致（传全 0 的层级或 None 都一样）。"""
        ei, level = build_synth(with_extras=False)
        for seed in SEEDS:
            tr, va, te, trk, vak, tek = legacy_split(ei, seed)
            for lv in (level, None):
                sp = split_edges(ei, lv, 12, seed, 'kinase')
                self.assertTrue(torch.equal(sp['train'], tr) and torch.equal(sp['val'], va)
                                and torch.equal(sp['test'], te), f'seed={seed}, 有层级标注={lv is not None}')
                self.assertTrue(np.array_equal(sp['train_k'], trk) and np.array_equal(sp['val_k'], vak)
                                and np.array_equal(sp['test_k'], tek))
                self.assertEqual((sp['n_family'], sp['n_auto']), (0, 0))
            self.assertIsNone(split_edges(ei, None, 12, seed, 'kinase')['train_level'])
            tl = split_edges(ei, level, 12, seed, 'kinase')['train_level']
            self.assertEqual(tl.numel(), tr.size(1))
            self.assertTrue(torch.equal(tl, torch.zeros_like(tl)))

    def test_split_edge_mode(self):
        sp = split_edges(self.ei, self.level, 12, 42, 'edge')
        tr_s, va, te = keys_of(sp['train_specific']), keys_of(sp['val']), keys_of(sp['test'])
        # 旧协议：特定激酶的边随机划分，家族层面边与全部自磷酸化边都进训练图
        self.assertEqual(keys_of(sp['train']), tr_s | self.fam_keys | self.auto_keys)
        self.assertEqual(tr_s | va | te, self.spec_keys)
        self.assertFalse(tr_s & va or tr_s & te or va & te)
        self.assertEqual((sp['n_family'], sp['n_auto']), (len(self.fam_keys), len(self.auto_keys)))
        eval_names = {k for k, _ in va | te}
        self.assertTrue(all(bool(sp['eval_kinase'][k]) == (k in eval_names) for k in range(12)))

    def test_family_context(self):
        kinase_sites, site_kinases = true_maps(self.ei)
        before = {k: set(v) for k, v in kinase_sites.items()}
        keys, mask = apply_family_context(kinase_sites, site_kinases, FAMILY_MEMBERS, NUM_SITE)
        # 家族节点的位点 = 家族层面位点 ∪ 成员自己的特异位点
        self.assertEqual(kinase_sites[10], {0, 1, 2, 3, 4, 13, 15, 17})
        self.assertEqual(kinase_sites[11], {3, 4, 5, 12, 14})
        # 成员的位点并入所属家族的家族层面位点；激酶 2 属于两个家族，两份都并入
        self.assertEqual(kinase_sites[0], {0, 1, 13, 15, 17})
        self.assertEqual(kinase_sites[1], {1, 2, 13})
        self.assertEqual(kinase_sites[2], {1, 3, 4, 5, 13, 14})
        self.assertEqual(kinase_sites[3], {5, 14})
        self.assertEqual(kinase_sites[9], {12, 14})
        # 不属于任何家族的激酶不变
        self.assertEqual(kinase_sites[4], before[4])
        self.assertEqual(kinase_sites[5], before[5])
        # 只读补全之前的位点集合：家族 B 不会把家族 A 的家族层面位点（1、13）当成成员位点吸收
        self.assertNotIn(1, kinase_sites[11])
        self.assertNotIn(13, kinase_sites[11])
        # 位点 -> 激酶：家族层面位点的激酶集合并入成员；成员位点的激酶集合并入所属家族节点
        self.assertEqual(site_kinases[13], {0, 1, 2, 10})
        self.assertEqual(site_kinases[14], {2, 3, 9, 11})
        self.assertEqual(site_kinases[5], {2, 3, 9, 11})
        self.assertEqual(site_kinases[1], {0, 1, 2, 10})
        self.assertEqual(site_kinases[3], {2, 10, 11})
        self.assertEqual(site_kinases[12], {9, 11})
        self.assertEqual(site_kinases[15], {0, 10})
        self.assertEqual(site_kinases[6], {4})        # 与任何家族无关的位点不变
        # extra_keys = 上面新补的 (激酶, 位点) 对：排序、不重复、与集合形式一致
        want = ({(10, s) for s in (0, 1, 2, 3, 4, 15, 17)}
                | {(11, s) for s in (3, 4, 5, 12)}
                | {(m, s) for m in (0, 1, 2) for s in (1, 13)}
                | {(m, s) for m in (2, 3, 9) for s in (5, 14)})
        self.assertEqual(keys.dtype, np.int64)
        self.assertEqual({(int(x) // NUM_SITE, int(x) % NUM_SITE) for x in keys}, want)
        self.assertEqual(len(keys), len(want))
        self.assertTrue(np.all(np.diff(keys) > 0))
        # 评估屏蔽：成员移出所属家族的家族层面位点；家族节点与无关激酶为 None
        self.assertEqual(len(mask), 12)
        self.assertEqual([list(x) if x is not None else None for x in mask],
                         [[1, 13], [1, 13], [1, 5, 13, 14], [5, 14], None, None, None, None, None, [5, 14], None, None])

    def test_family_context_old_graph(self):
        kinase_sites, site_kinases = true_maps(self.ei)
        before = ({k: set(v) for k, v in kinase_sites.items()}, {s: set(v) for s, v in site_kinases.items()})
        keys, mask = apply_family_context(kinase_sites, site_kinases, None, NUM_SITE)
        self.assertEqual((keys.shape, list(mask)), ((0,), []))
        self.assertEqual(({k: set(v) for k, v in kinase_sites.items()},
                          {s: set(v) for s, v in site_kinases.items()}), before)

    def test_no_negative_hits_family_level_sites(self):
        """用训练脚本同样的过滤生成负样本：不撞任何已知边，也不撞所属家族的家族层面位点。"""
        kinase_sites, site_kinases = true_maps(self.ei)
        _, mask = apply_family_context(kinase_sites, site_kinases, FAMILY_MEMBERS, NUM_SITE)
        for k in range(10):
            neg = corrupt(np.full(2000, k), NUM_SITE, kinase_sites)     # tail：保留激酶换位点
            self.assertFalse(any(int(s) in kinase_sites[k] for s in neg), f'激酶 {k} 的负样本撞上已知边')
            masked = masked_sites(mask, k)
            if masked is not None:
                self.assertFalse(set(neg.tolist()) & set(masked.tolist()), f'激酶 {k} 的负样本撞上家族层面位点')
        for s in range(18):
            neg = corrupt(np.full(2000, s), 12, site_kinases)          # head：保留位点换激酶
            self.assertFalse(any(int(k) in site_kinases[s] for k in neg), f'位点 {s} 的负样本撞上已知边')

    def test_mask_family_sites(self):
        scores = torch.arange(NUM_SITE, dtype=torch.float32)
        masked = np.array([1, 13, 15], dtype=np.int64)
        n = mask_family_sites(scores, [15], masked)              # 位点 15 是该激酶自己的特异正例，不屏蔽
        self.assertEqual(n, 2)
        self.assertEqual(float(scores[1]), float('-inf'))
        self.assertEqual(float(scores[13]), float('-inf'))
        self.assertTrue(torch.isfinite(scores[0]) and torch.isfinite(scores[14]) and torch.isfinite(scores[15]))
        self.assertEqual(mask_family_sites(torch.zeros(NUM_SITE), [1, 13, 15], masked), 0)   # 全是正例
        self.assertEqual(mask_family_sites(torch.zeros(NUM_SITE), [], None), 0)
        self.assertEqual(mask_family_sites(torch.zeros(NUM_SITE), [], np.array([], dtype=np.int64)), 0)

    def test_masked_sites(self):
        self.assertIsNone(masked_sites(None, 3))
        self.assertIsNone(masked_sites([], 3))                  # 旧图：apply_family_context 返回空列表
        arr = np.array([1, 13], dtype=np.int64)
        self.assertIs(masked_sites([None, arr], 1), arr)
        self.assertIsNone(masked_sites([None, arr], 0))

    def test_apply_train_edges(self):
        sp = split_edges(self.ei, self.level, 12, 42, 'kinase')
        g = _Graph()
        g[PHOS_KEY].edge_index = self.ei
        g[REV_KEY].edge_index = self.ei[[1, 0]]
        g[PHOS_KEY].edge_level = self.level
        g[REV_KEY].edge_level = self.level
        apply_train_edges(g, sp['train'], sp['train_level'])
        self.assertTrue(torch.equal(g[PHOS_KEY].edge_index, sp['train']))
        self.assertTrue(torch.equal(g[REV_KEY].edge_index, sp['train'][[1, 0]]))
        self.assertTrue(torch.equal(g[PHOS_KEY].edge_level, sp['train_level']))
        self.assertTrue(torch.equal(g[REV_KEY].edge_level, sp['train_level']))
        self.assertEqual(g[PHOS_KEY].edge_level.numel(), g[PHOS_KEY].edge_index.size(1))
        # 层级标注与训练边逐条对应；评估激酶的自磷酸化边被排除，其余并入训练快照
        expect_level = sp['train_level'].tolist()
        kept_auto = {k for k, _ in AUTOPHOS if not bool(sp['eval_kinase'][k])}
        self.assertEqual(sorted(expect_level), [0] * sp['train_specific'].size(1) + [1] * len(FAMILY) + [2] * len(kept_auto))
        # 旧图：只换边，不写层级标注
        tr = legacy_split(self.ei, 42)[0]
        g2 = _Graph()
        g2[PHOS_KEY].edge_index = self.ei
        g2[REV_KEY].edge_index = self.ei[[1, 0]]
        apply_train_edges(g2, tr, None)
        self.assertTrue(torch.equal(g2[PHOS_KEY].edge_index, tr))
        self.assertNotIn('edge_level', g2[PHOS_KEY])


class TestRealGraph(unittest.TestCase):
    """B. 真实图上的不变量。"""

    def test_real_graph(self):
        if not os.path.exists(GRAPH_PATH):
            raise unittest.SkipTest(f'找不到图文件 {GRAPH_PATH}；用参数指定图路径')
        data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
        ei = data['kinase', 'phosphorylates', 'site'].edge_index
        names = list(data['kinase'].node_names)
        num_kinase, num_site = len(names), data['site'].x.size(0)
        level, members = read_edge_levels(data)
        pairs = list(zip(ei[0].tolist(), ei[1].tolist()))

        if level is None:
            # ---- 旧图：与原划分逐边一致
            print(f'\n[旧图] {GRAPH_PATH}: 激酶 {num_kinase}，位点 {num_site}，边 {ei.size(1)}')
            for seed in SEEDS:
                tr, va, te, trk, vak, tek = legacy_split(ei, seed)
                sp = split_edges(ei, level, num_kinase, seed, 'kinase')
                self.assertTrue(torch.equal(sp['train'], tr) and torch.equal(sp['val'], va)
                                and torch.equal(sp['test'], te), f'seed={seed}: 新划分与原代码不一致')
                self.assertTrue(np.array_equal(sp['train_k'], trk) and np.array_equal(sp['val_k'], vak)
                                and np.array_equal(sp['test_k'], tek), f'seed={seed}: 激酶归属不一致')
                self.assertIsNone(sp['train_level'])
                self.assertEqual((sp['n_family'], sp['n_auto']), (0, 0))
            sp = split_edges(ei, level, num_kinase, 42, 'kinase')
            sizes = (sp['train'].size(1), sp['val'].size(1), sp['test'].size(1))
            print(f'[旧图] {len(SEEDS)} 个种子下与原划分逐边一致；seed=42 的划分 '
                  f'{sizes[0]} / {sizes[1]} / {sizes[2]}')
            if num_kinase == OLD_GRAPH_NODES:
                self.assertEqual(sizes, OLD_GRAPH_SPLIT)
            return

        # ---- 新图：家族节点标注
        self.assertEqual(level.numel(), ei.size(1))
        self.assertEqual(len(members), num_kinase)
        self.assertEqual(len(set(pairs)), len(pairs), '磷酸化边有重复')
        lv = level.tolist()
        is_family = np.zeros(num_kinase, dtype=bool)
        is_family[np.asarray(data['kinase'].is_family).astype(bool)] = True
        self.assertTrue(np.array_equal(is_family, np.array([n.startswith('FAMILY_') for n in names])),
                        'is_family 标注与 FAMILY_ 节点名不一致')
        fam_idx = np.where(is_family)[0].tolist()
        self.assertTrue(fam_idx, '新图里没有家族节点')
        fams_of = defaultdict(list)
        for f in fam_idx:
            self.assertTrue(members[f], f'{names[f]} 没有成员')
            self.assertTrue(all(0 <= int(j) < num_kinase and not is_family[j] for j in members[f]),
                            f'{names[f]} 的成员不合法')
            for j in members[f]:
                fams_of[j].append(f)
        self.assertTrue(np.array_equal(is_family[ei[0].numpy()], np.asarray(lv) == 1),
                        'level 1 的边应当恰好是从家族节点出发的边')

        # 新图：特定激酶的边与原划分逐边一致
        spec = ei[:, level == 0]
        for seed in SEEDS:
            tr, va, te, trk, vak, tek = legacy_split(spec, seed)
            sp = split_edges(ei, level, num_kinase, seed, 'kinase')
            self.assertTrue(torch.equal(sp['train_specific'], tr) and torch.equal(sp['val'], va)
                            and torch.equal(sp['test'], te), f'seed={seed}: 特定激酶的边与原代码不一致')
            self.assertTrue(np.array_equal(sp['train_k'], trk) and np.array_equal(sp['val_k'], vak)
                            and np.array_equal(sp['test_k'], tek), f'seed={seed}: 激酶归属不一致')

        # 新图：训练图的组成
        sp = split_edges(ei, level, num_kinase, 42, 'kinase')
        train_pairs = list(zip(sp['train'][0].tolist(), sp['train'][1].tolist()))
        train, val, test = set(train_pairs), keys_of(sp['val']), keys_of(sp['test'])
        val_k, test_k = set(sp['val_k'].tolist()), set(sp['test_k'].tolist())
        pair_level = dict(zip(pairs, lv))
        fam_keys = {p for p, l in pair_level.items() if l == 1}
        auto_keys = {p for p, l in pair_level.items() if l == 2}
        kept_auto = {p for p in auto_keys if p[0] not in (val_k | test_k)}
        self.assertFalse(train & val or train & test or val & test)
        self.assertTrue(all(pair_level[p] == 0 for p in val | test), '验证 / 测试集里出现了非特定激酶的边')
        self.assertTrue(all(k in val_k for k, _ in val) and all(k in test_k for k, _ in test))
        self.assertFalse(any(k in (val_k | test_k) for k, _ in keys_of(sp['train_specific'])),
                         '评估激酶的特定激酶边出现在训练图里')
        self.assertEqual(train, keys_of(sp['train_specific']) | fam_keys | kept_auto)
        self.assertEqual((sp['n_family'], sp['n_auto']), (len(fam_keys), len(kept_auto)))
        self.assertEqual(sp['train_level'].tolist(), [pair_level[p] for p in train_pairs])

        # 新图：家族层面位点对成员既不是正例也不是负样本
        kinase_sites, site_kinases = true_maps(ei)
        fam_sites = defaultdict(set)
        for (f, s), l in pair_level.items():
            if l == 1:
                fam_sites[f].add(s)
        _, mask = apply_family_context(kinase_sites, site_kinases, members, num_site)
        for f in fam_idx:
            for m in members[f]:
                self.assertTrue(fam_sites[f] <= kinase_sites[m], f'{names[m]} 的位点集合缺少 {names[f]} 的家族层面位点')
                if fam_sites[f]:
                    self.assertIsNotNone(mask[m])
                    self.assertTrue(fam_sites[f] <= set(mask[m].tolist()), f'{names[m]} 的评估屏蔽缺少家族层面位点')
        for (m, s), l in pair_level.items():
            if l == 0:
                self.assertTrue(all(f in site_kinases[s] for f in fams_of[m]),
                                f'位点 {s} 的激酶集合缺少 {names[m]} 所属的家族节点')
        n_spec = len(pair_level) - len(fam_keys) - len(auto_keys)
        print(f'\n[新图] {GRAPH_PATH}: 激酶 {num_kinase}（其中家族节点 {len(fam_idx)}），位点 {num_site}，边 {len(pairs)}'
              f'（特定激酶 {n_spec}，家族层面 {len(fam_keys)}，自磷酸化 {len(auto_keys)}）')
        print(f'[新图] seed=42 划分: 训练边 {len(train)}（特定激酶 {len(keys_of(sp["train_specific"]))} + 家族层面 '
              f'{len(fam_keys)} + 自磷酸化 {len(kept_auto)}）/ 验证 {len(val)} / 测试 {len(test)}；'
              f'评估激酶 {len(val_k | test_k)} 个')
        print('[新图] 家族层面位点对成员的负样本过滤与评估屏蔽：通过')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data_prep/edge_levels.py 的测试')
    parser.add_argument('graph', nargs='?', default=DEFAULT_GRAPH,
                        help='异构图路径（默认 origin_data/joint_hetero_graph.pt）')
    args, remaining = parser.parse_known_args()
    GRAPH_PATH = args.graph
    if not os.path.exists(GRAPH_PATH):
        print(f'提示: 找不到图文件 {GRAPH_PATH}，真实图用例将跳过（只跑合成图用例）')
    unittest.main(argv=[sys.argv[0]] + remaining)
