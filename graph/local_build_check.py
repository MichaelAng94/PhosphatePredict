#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地检查建图脚本的激酶名归一（无需 torch：用替身模块顶替 torch / PyG / node2vec，
端到端运行真实的 graph/build_joint_graph.py，图对象只写到临时目录）。

用途：改了别名表 / 人工对照表 / 建图脚本之后，在没有 GPU 的机器上确认：
  T1 查表器与两种命名语义（KinBase / 现用符号）的行为；
  T2 同一规范名有多个特征文件时的取舍；
  T3 测试环境可信——按旧行为（只大写归一）能复现老图的已知数字
     （567 节点、453 个有边、11123 位点、唯一边 11601/2935/1680）；
  T4 新图满足：节点名是规范名、没有泛称 / 非蛋白激酶节点、边无重复、
     没有序列完全相同的两个节点，并预览 cold-start 划分的组成。
底物特征用占位向量代替，所以只验证名字与图结构，不验证特征数值。

用法（项目根目录，只需 numpy / pandas / scikit-learn）:
    python3 graph/local_build_check.py
"""

import importlib.util
from pathlib import Path
import os
import pickle
import sys
import tempfile
import types
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)


class Arr(np.ndarray):
    """够用的 torch 张量替身：t() / contiguous() / size(i)。"""

    def __new__(cls, x, dtype=None):
        a = np.asarray(x)
        if dtype == 'int64':
            a = a.astype(np.int64)
        elif dtype == 'float32':
            a = a.astype(np.float32)
        return a.view(cls)

    def __array_finalize__(self, obj):
        pass

    def t(self):
        return self.transpose().view(Arr)

    def contiguous(self):
        return self

    def size(self, *idx):
        return self.shape[idx[0]] if idx else int(np.prod(self.shape))


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_module('torch', tensor=lambda x, dtype=None: Arr(x, dtype), float='float32', long='int64',
        zeros=lambda *shape, dtype=None: Arr(np.zeros(shape), dtype),
        empty=lambda shape, dtype=None: Arr(np.empty(shape, np.int64), dtype),
        stack=lambda xs, dim=0: Arr(np.stack([np.asarray(x) for x in xs], axis=dim)),
        save=lambda obj, path: pickle.dump(obj, open(path, 'wb')))


class _NS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class HeteroData(dict):
    def __getitem__(self, key):
        return super().setdefault(key, _NS())

    def __str__(self):
        return (f"HeteroData(节点类型={[k for k in self if not isinstance(k, tuple)]}, "
                f"边类型={[k for k in self if isinstance(k, tuple)]})")


_module('torch_geometric')
_module('torch_geometric.data', HeteroData=HeteroData)
_module('node2vec', Node2Vec=type('Node2Vec', (), {'__init__': lambda self, *a, **k: None}))
_module('networkx', Graph=type('Graph', (), {'__init__': lambda self: None, 'add_edge': lambda self, a, b: None}))

spec = importlib.util.spec_from_file_location('bjg', f'{ROOT}/graph/build_joint_graph.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from data_prep.kinase_names import EXCLUDED_ENTITY_TYPES, KinaseNamer, load_family_table

NAMER = KinaseNamer.load()
fails = []


def check(cond, msg):
    print(("  [通过] " if cond else "  [失败] ") + msg)
    if not cond:
        fails.append(msg)


# ---------------- T1 查表器 ----------------
print("=== T1 查表器（kinase_names.KinaseNamer）===")
for raw, canon, etype in [('cdc2', 'CDK1', 'kinase'), ('FRAP', 'MTOR', 'kinase'), ('PKACA', 'PRKACA', 'kinase'),
                          ('P38A(MAPK14)', 'MAPK14', 'kinase'), ('PKC_DELTA', 'PRKCD', 'kinase'),
                          ('CAM-KII_ALPHA', 'CAMK2A', 'kinase'), ('PKA_GROUP', 'PKA_GROUP', 'group'),
                          ('AUTOCATALYSIS', 'AUTOCATALYSIS', 'other'), ('GUK1', 'GUK1', 'non_kinase'),
                          ('cdc2_drome', 'CDC2_DROME', 'non_human'), ('XYZ_NOT_A_KINASE', 'XYZ_NOT_A_KINASE', 'unmapped')]:
    c, e = NAMER.info(raw)
    check(c == canon and e == etype, f"{raw} -> {c}（{e}）")
for raw, kc, sc in [('GCK', 'MAP4K2', 'GCK'), ('MST1', 'STK4', 'MST1'), ('PDK1', 'PDPK1', 'PDK1'),
                    ('PRKACA', 'PRKACA', 'PRKACA'), ('PKD1', 'PRKD1', 'PKD1')]:
    ck, ek = NAMER.info(raw, semantics='kinbase')
    cs, es = NAMER.info(raw, semantics='symbol')
    check(ck == kc and ek == 'kinase' and cs == sc, f"两种语义 {raw}: kinbase -> {ck}（{ek}）| symbol -> {cs}（{es}）")
check(NAMER.canonical('GCK', semantics='symbol') == 'GCK'
      and m.canonical_kinase_name('GCK', semantics='symbol') == ''
      and m.canonical_kinase_name('GCK') == 'MAP4K2', "建图脚本：GCK 在 PPI 语义下被排除、在 KSR 语义下归到 MAP4K2")

# ---------------- T2 特征文件取舍 ----------------
print("\n=== T2 同一规范名有多个特征文件时的取舍 ===")
fa = {}
for p in m.KINASE_FASTAS:
    n, b = None, []
    for line in open(p):
        line = line.strip()
        if line.startswith('>'):
            if n:
                fa.setdefault(n, ''.join(b))
            n, b = line[1:].split()[0].strip().upper(), []
        elif line:
            b.append(line)
    if n:
        fa.setdefault(n, ''.join(b))
order = sorted(fa)
code = {n: i for i, n in enumerate(order)}
tmp = tempfile.mkdtemp(prefix='pp_npy_')
for i, n in enumerate(order):
    v = np.full(16, code[n], dtype=np.float32)
    np.save(os.path.join(tmp, f'{i}.npy'), {'name': n, 'mean': {34: v, 35: v, 36: v}}, allow_pickle=True)
m.KINASE_FEATURE_DIRS = [tmp]
feats = m.load_kinase_features_esm()
src = {c: order[int(np.asarray(v).ravel()[0])] for c, v in feats.items()}
for canon, expect in [('CDK1', 'CDK1'), ('PRKACA', 'PRKACA'), ('PDPK1', 'PDK1'), ('STK4', 'MST1'), ('SBK1', 'SBK'),
                      ('DMPK', 'DMPK1'), ('MAP4K2', 'MAP4K2'), ('OXSR1', 'OSR1'), ('GRK7', 'GRK7')]:
    check(src.get(canon) == expect, f"{canon} 的特征取自 {src.get(canon)}（期望 {expect}）")
check('GCK' not in feats and 'SDK1' not in feats and 'MST1' not in feats, "UniProt 来源的 SDK1（sidekick-1）等不会被当作激酶特征")
feats2 = m.load_kinase_features_esm()
check({c: int(np.asarray(v).ravel()[0]) for c, v in feats.items()} == {c: int(np.asarray(v).ravel()[0]) for c, v in feats2.items()},
      "两次加载结果一致（不依赖目录遍历顺序）")

# ---------------- T3 端到端建图 ----------------
ks = pd.read_csv(f'{ROOT}/origin_data/kinase_substrate_pairs_detailed.csv', dtype=str).fillna('')
site_ids = set(ks.EPSD_ID.str.strip() + '_' + ks.Position.str.strip())
prots = {s.rsplit('_', 1)[0] for s in site_ids}
m.load_site_features = lambda: {s: Arr(np.ones(8, np.float32)) for s in site_ids}
m.load_substrate_protein_features = lambda: {p: Arr(np.ones(16, np.float32)) for p in prots}
ppi_nodes = set(pd.read_csv(m.PPI_CSV, dtype=str).preferredName_A.str.upper()) | set(
    pd.read_csv(m.PPI_CSV, dtype=str).preferredName_B.str.upper())
m.generate_kinase_features_from_ppi = lambda csv, kl: {k: Arr(np.full(128, 7.0, np.float32)) for k in kl if k in ppi_nodes}
m.OUTPUT_GRAPH = os.path.join(tmp, 'graph.pt')
new_canon = m.canonical_kinase_name


def summarize(tag, data):
    names = list(data['kinase'].node_names)
    ei = data['kinase', 'phosphorylates', 'site'].edge_index
    kidx, sidx = np.asarray(ei[0]), np.asarray(ei[1])
    sites = list(data['site'].node_names)
    kn = [names[i] for i in kidx]
    kwe = np.array(sorted(set(kn)))
    tr, tmpk = train_test_split(kwe, test_size=0.3, random_state=42)
    va, te = train_test_split(tmpk, test_size=1/3, random_state=42)
    npos = Counter(kn)
    uniq = len(set(zip(kn, sidx)))
    print(f"  [{tag}] 激酶节点 {len(names)}，其中有磷酸化边的 {len(kwe)}；位点 {len(sites)}；"
          f"磷酸化边 {len(kn)}（唯一 {uniq}）；PPI 边 {np.asarray(data['kinase', 'ppis', 'kinase'].edge_index).shape[1]}")
    print(f"       划分(seed=42): 训练 {len(tr)} 激酶 / 测试 {len(te)} / 验证 {len(va)}；"
          f"唯一边 训练 {sum(npos[k] for k in tr)} / 验证 {sum(npos[k] for k in va)} / 测试 {sum(npos[k] for k in te)}")
    return names, kn, sidx, tr, va, te, npos


print("\n=== T3a 旧行为（只大写归一）——应复现老图：567 节点、453 有边、11123 位点、唯一边 11601/2935/1680 ===")
m.canonical_kinase_name = lambda raw, semantics='kinbase': m.normalize_kinase_name(raw)
old = summarize('旧', m.build_joint_graph())

print("\n=== T3b 新行为（别名表归一）===")
m.canonical_kinase_name = new_canon
data = m.build_joint_graph()
names, kn, sidx, tr, va, te, npos = summarize('新', data)
sites = list(data['site'].node_names)

check(names == sorted(names), "节点名已排序")
# 节点名是现用符号，按符号语义核对（'PDK1' 节点是 HGNC 的丙酮酸脱氢酶激酶 1，KSR 的 PDHK1 归到它；
# KinBase 语义下字符串 'PDK1' 指 PDPK1，二者是同一个节点的两种来源写法之外的另一个蛋白）
bad = [(n, NAMER.canonical(n, 'symbol'), NAMER.entity_type(n, 'symbol')) for n in names
       if NAMER.entity_type(n, 'symbol') == 'kinase' and NAMER.canonical(n, 'symbol') != n]
check(not bad, f"kinase 类型节点名都是规范名（例外: {bad[:8]}）")
check(all(NAMER.entity_type(n) not in EXCLUDED_ENTITY_TYPES for n in names), "图中没有泛称 / 非蛋白激酶 / 非人物种 / other 节点")
check(len(set(zip(kn, sidx))) == len(kn), "磷酸化边无重复")
check(set(sidx.tolist()) == set(range(len(sites))), "位点索引连续")
pe = data['kinase', 'ppis', 'kinase'].edge_index
check(np.asarray(pe).shape[1] == 0 or (np.asarray(pe[0]).max() < len(names) and np.asarray(pe[1]).max() < len(names)),
      "PPI 边两端都在节点集内")

# 同序列双胞胎
seq_of = {}
for n in names:
    for a in NAMER.aliases(n):
        if a in fa:
            seq_of.setdefault(n, fa[a])
byseq = defaultdict(list)
for n, s in seq_of.items():
    byseq[s].append(n)
twins = [sorted(v) for v in byseq.values() if len(v) > 1]
check(not twins, f"图中没有序列完全相同的两个节点（{len(twins)} 组）" + (f"：{twins[:5]}" if twins else ""))

# 旧测试集 4 个泄漏激酶的去向
old_test = ['KHS1', 'SDK1', 'TNNI3K', 'GUK1', 'PRKAR2A', 'CDK11B', 'PITSLRE', 'NUAK2', 'WEE2', 'NEK3', 'BARK2', 'TYRO3',
            'FLT1', 'CAMK2G', 'CAMLCK', 'CHAK2', 'PAK6', 'BRSK2', 'PASK', 'TAF1', 'KDR', 'ABL2', 'PKM', 'RIPK3', 'CAMKK2',
            'AURC', 'P38D', 'RON', 'GRK3', 'RSK1', 'VRK1', 'OSR1', 'GRK5', 'MELK', 'CK1E', 'LKB1', 'GRK2', 'TTK', 'CK1D',
            'EGFR', 'AMPKA1', 'MTOR', 'CK2A1']
part = {n: s for s, grp in (('train', tr), ('val', va), ('test', te)) for n in grp}   # 激酶名 -> 所属划分
print("  旧测试集 46 个激酶在新划分中的归属:")
gone = [k for k in old_test if (NAMER.canonical(k) not in names) or NAMER.entity_type(k) != 'kinase']
print("    不再是节点（被排除）:", gone)
print("    归入:", dict(Counter(part.get(NAMER.canonical(k), '-') for k in old_test if k not in gone)))
print("    仍在测试集的:", sorted(NAMER.canonical(k) for k in old_test if k not in gone and part.get(NAMER.canonical(k)) == 'test'))

fam = load_family_table()
lab = dict(zip(fam.Name, fam.Family))
with_edges = sorted(set(kn))
cov = np.mean([n in lab for n in with_edges])
print(f"\n  家族标签覆盖: 有边的激酶 {len(with_edges)} 个中 {sum(n in lab for n in with_edges)} 个有标签（{cov:.1%}），"
      f"涉及 {len({lab[n] for n in with_edges if n in lab})} 个家族")
for nm, s in (('训练', tr), ('验证', va), ('测试', te)):
    b = np.array([npos[k] for k in s])
    top = max(b) / b.sum()
    print(f"  {nm}集 {len(s)} 个激酶: 底物数中位 {np.median(b):.0f}，仅 1 条的 {int((b == 1).sum())} 个，"
          f"<=5 条的 {int((b <= 5).sum())} 个，>50 条的 {int((b > 50).sum())} 个；最大单激酶占该集边数 {top:.0%}"
          f"（{max(s, key=lambda k: npos[k])}）")
print("  测试集全部激酶（按底物数）:", ", ".join(f"{k}:{npos[k]}" for k in sorted(te, key=lambda k: -npos[k])))
ent = Counter(NAMER.entity_type(n) for n in names)
print("  节点实体类型:", dict(ent), "| 其中 unmapped/ambiguous 节点:",
      [n for n in names if NAMER.entity_type(n) in ('unmapped', 'ambiguous') and n in npos])

print("\n结果:", "全部通过" if not fails else f"{len(fails)} 项失败: {fails}")
