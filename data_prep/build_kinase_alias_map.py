#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成激酶别名映射表与家族表：合并同一激酶的多个名字，并按现用基因名重做家族标签。

问题：KSR / PPI / ESM 特征里的激酶名混用旧名与现用名（CDC2 与 CDK1、FRAP 与 MTOR、
PKACA 与 PRKACA…），只做大写归一会把同一个蛋白拆成多个节点——训练标签被拆散，
冷启动评估里还会出现"测试激酶的同序列别名在训练集"的泄漏。

做法：规范名 = HGNC 现用基因符号。

(1) Manning 名 -> 现用符号（得到"人蛋白激酶成员"与 Group/Family/Subfamily 分类）
      人工对照表 -> 名称索引（现用符号 / 旧符号 / 别名；多命中不猜）-> "名/别名"拆分
      -> 与 FASTA 中已解析名字的序列完全相同
    注意：HGNC 基因分组里没有覆盖全部蛋白激酶的总组，所以"是不是蛋白激酶"只认 Manning
    映射结果 + 人工对照表。Manning 名恰好等于另一个蛋白的现用符号的情况（PDK1、GCK、
    MYT1、PKD1、MST1、ACTR2、OSR1、SRM）靠人工对照表纠正。
(2) 原始名 -> 规范名，按证据强度依次尝试：
      0 人工对照表 kinase_alias_manual.csv
      1 现用符号 / 2 旧符号 / 3 别名（别名命中多个基因且只有一个是激酶时取该激酶）
      4 '名(符号)'、'名 ISO2'、'名/别名' 等 Phospho.ELM 写法
      5 以 _GROUP 结尾的泛称
      6 Manning 名（Phospho.ELM 用 KinBase 命名）
      7 忽略标点的键匹配 + 希腊字母替换（PKC_ALPHA -> PKCA）+ CAMK 前缀归一
      8 Manning 家族 / 亚家族 / 大类名（CDK、PAK、CAMK…）
      9 UniProt accession 一致（fetch_report.csv 对 HGNC uniprot_ids）
      10 与已解析名字的激酶序列完全相同

实体类型：kinase / group / non_kinase / non_human / other / ambiguous / unmapped。
  kinase      具体的人蛋白激酶
  group       家族/泛称（PKA、MAPK、XX_GROUP… 的注释无法归属到某一个激酶）
  non_kinase  能对应到基因但不是蛋白激酶（GUK1、PRKAR2A、PKM、CK2B…）
  non_human   带非人物种后缀（_BOVIN、_DROME…）
  other       融合蛋白、自磷酸化、注释文字等
  ambiguous   别名同时命中多个不同家族的激酶
  unmapped    以上都解析不到

输入（origin_data/）：HGNC 完整表、kinase_family.xls（Manning 2002）、kinase_alias_manual.csv、
三个激酶序列 FASTA、fetch_report.csv、kinase_substrate_pairs_detailed.csv，以及 PPI 表。
输出：origin_data/kinase_alias_map.csv、origin_data/kinase_family.csv

用法（项目根目录）:
    python3 data_prep/build_kinase_alias_map.py [--hgnc <HGNC 表>] [--out-alias <csv>] [--out-family <csv>]
HGNC 完整表（约 16 MB）下载:
    curl -L -o origin_data/hgnc_complete_set.txt \\
      https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "origin_data"
HGNC_DEFAULT = DATA / "hgnc_complete_set.txt"
MANNING_XLS = DATA / "kinase_family.xls"
MANUAL_CSV = DATA / "kinase_alias_manual.csv"   # raw_name, canonical, entity_type, group, family, subfamily, note
KSR_CSV = DATA / "kinase_substrate_pairs_detailed.csv"
PPI_CSV = PROJECT_ROOT / "PPI/kinase_ppi_network.csv"
FETCH_REPORT = DATA / "fetch_report.csv"
FASTAS = [DATA / "kinase_sequences.fasta",
          DATA / "kinase_sequences_uniprot.fasta",
          DATA / "kinase_sequences_out.fasta"]
# 按基因名从 UniProt 检索得到的序列（fetch_kinase_sequences_uniprot.py）：其名字是现用符号语义
UNIPROT_FASTA = DATA / "kinase_sequences_uniprot.fasta"
OUT_ALIAS = DATA / "kinase_alias_map.csv"
OUT_FAMILY = DATA / "kinase_family.csv"

# Swiss-Prot 风格物种码。_HUMAN 去后缀后按人源处理；其余物种不并入人激酶。
HUMAN_SUFFIX = "HUMAN"
NONHUMAN_SUFFIXES = {
    "BOVIN", "MOUSE", "RAT", "RABIT", "RABBIT", "CHICK", "XENLA", "XENLU", "DROME", "CAEEL",
    "SCHPO", "YEAST", "PIG", "SHEEP", "CANFA", "ARATH", "HSV", "HCMV", "SV40", "VACCV", "EBVB",
}
SPECIES_RE = re.compile(r"^(.+)_([A-Z0-9]+)$")
HGNC_REQUIRED = ("symbol", "alias_symbol", "prev_symbol", "uniprot_ids", "locus_group")
# KinBase 用单个字母表示 Greek letter：PKCA/alpha、PKCD/delta、P38A/alpha …
GREEK = {"ALPHA": "A", "BETA": "B", "GAMMA": "G", "DELTA": "D", "EPSILON": "E",
         "ZETA": "Z", "ETA": "H", "THETA": "T", "IOTA": "I"}
PUNCT_RE = re.compile(r"[\s_\-\.]+")
GROUP_SUFFIX_RE = re.compile(r"_GROUP$")
PAREN_RE = re.compile(r"^(.+?)\(([A-Z0-9][A-Z0-9\-]*)\)$")
ISO_RE = re.compile(r"^(.+?)\s+ISO\d+$")


def upper(raw) -> str:
    return raw.strip().upper() if isinstance(raw, str) else ""


def split_multi(value) -> list:
    """HGNC 的多值字段以 '|' 分隔。"""
    return [x.strip().upper() for x in str(value).split("|") if x.strip()]


def compact(name) -> str:
    """忽略空格 / 下划线 / 连字符 / 点：'PKC_ALPHA' 与 'PKC-ALPHA' 得到同一个键。"""
    return PUNCT_RE.sub("", name)


def read_hgnc(path):
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False).fillna("")
    missing = [c for c in HGNC_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"HGNC 表缺少列 {missing}，实际列: {list(df.columns)[:30]}…")
    return df


def read_manning(path):
    df = pd.read_excel(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    need = ("Name", "Group", "Family", "Subfamily", "Pseudogene?", "Protein")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Manning 表缺少列 {missing}，实际列: {list(df.columns)}")
    return df


def read_fasta(paths):
    """{大写名: 序列}；同名出现在多个文件时取首个。"""
    return _read_fasta(paths, with_seq=True)


def fasta_names(paths):
    """FASTA 表头名集合（大写），不读序列。"""
    return set(_read_fasta(paths, with_seq=False))


def _read_fasta(paths, with_seq):
    seqs, name, buf = {}, None, []
    for path in paths:
        if not Path(path).exists():
            continue
        for line in open(path):
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs.setdefault(name, "".join(buf))
                name, buf = upper(line[1:].split()[0]), []
            elif line and with_seq:
                buf.append(line)
        if name is not None:
            seqs.setdefault(name, "".join(buf))
            name, buf = None, []
    return seqs


def strip_species(name):
    """返回 (去物种后缀的基础名, 物种码)；无已知物种后缀返回 (name, '')。"""
    m = SPECIES_RE.match(name)
    if m and m.group(2) in NONHUMAN_SUFFIXES | {HUMAN_SUFFIX}:
        return m.group(1), m.group(2)
    return name, ""


class HgncIndex:
    """HGNC 名称索引：符号 / 旧符号 / 别名 / UniProt accession -> 基因符号集合。"""

    def __init__(self, df):
        self.rows = {}
        self.prev, self.alias, self.uniprot = defaultdict(set), defaultdict(set), defaultdict(set)
        for r in df.itertuples(index=False):
            sym = upper(r.symbol)
            if not sym:
                continue
            self.rows[sym] = r
            for p in split_multi(getattr(r, "prev_symbol")):
                self.prev[p].add(sym)
            for a in split_multi(getattr(r, "alias_symbol")):
                if a != sym:
                    self.alias[a].add(sym)
            for acc in split_multi(getattr(r, "uniprot_ids")):
                self.uniprot[acc.split("-")[0]].add(sym)

    def is_protein_coding(self, sym):
        r = self.rows.get(sym)
        return bool(r) and str(r.locus_group) == "protein-coding gene"

    def resolve(self, name) -> set:
        """按名称索引解析（现用符号优先于旧符号、旧符号优先于别名），返回蛋白编码基因符号集合。"""
        if name in self.rows:
            return {name} if self.is_protein_coding(name) else set()
        if name in self.prev:
            return {c for c in self.prev[name] if self.is_protein_coding(c)}
        if name in self.alias:
            return {c for c in self.alias[name] if self.is_protein_coding(c)}
        return set()

    def source_of(self, name) -> str:
        return "symbol" if name in self.rows else "prev_symbol" if name in self.prev \
            else "alias_symbol" if name in self.alias else ""


def read_manual(path):
    """人工对照表：{大写原始名: 行}。"""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("canonical", "entity_type", "group", "family", "subfamily"):
        if col not in df.columns:
            df[col] = ""
    return {upper(r.raw_name): r for r in df.itertuples(index=False) if upper(r.raw_name)}


def manual_syms(row) -> set:
    """人工对照行给出的规范符号集合（多个用 ';' 分隔）。"""
    return {x for x in upper(row.canonical).split(";") if x}


def resolve_tokens(name, idx, manning_canon) -> set:
    """'HER2/ERBB2'、'PKG1/CGK-I' 这类"名/别名"写法：各段分别解析，返回并集。"""
    syms = set()
    for tok in name.split("/"):
        syms |= idx.resolve(tok) or set(manning_canon.get(tok, ()))
    return syms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hgnc", default=str(HGNC_DEFAULT), help="HGNC hgnc_complete_set.txt 路径")
    ap.add_argument("--out-alias", default=str(OUT_ALIAS), help="别名映射表输出路径")
    ap.add_argument("--out-family", default=str(OUT_FAMILY), help="家族表输出路径")
    args = ap.parse_args()
    out_alias, out_family = Path(args.out_alias), Path(args.out_family)

    idx = HgncIndex(read_hgnc(args.hgnc))
    manning = read_manning(MANNING_XLS)
    seqs = read_fasta(FASTAS)
    manual = read_manual(MANUAL_CSV)

    # ---------- Manning 条目 -> 现用符号（激酶成员 + 家群分类） ----------
    manning_families, manning_groups, manning_subfams = set(), set(), set()
    entries = []                                   # (Manning 名, Group, Family, Subfamily, 蛋白序列)
    for name, group, family, sub, pseudo, prot in zip(
            manning["Name"], manning["Group"], manning["Family"], manning["Subfamily"],
            manning["Pseudogene?"], manning["Protein"]):
        name, group, family, sub = upper(name), upper(group), upper(family), upper(sub)
        base, species = strip_species(name)
        if (not name) or upper(pseudo) == "Y" or (species and species != HUMAN_SUFFIX):
            continue                               # 伪基因 / 非人条目不进人激酶组
        manning_families |= {family} - {""}
        manning_groups |= {group} - {""}
        manning_subfams |= {sub} - {""}
        entries.append((base, group, family, sub, re.sub(r"\s+", "", str(prot)).upper()))

    manning_canon, pending = {}, []                # Manning 名 -> {现用符号…}
    for base, group, family, sub, prot in entries:
        syms = manual_syms(manual[base]) if base in manual else set()
        if not syms:
            syms = idx.resolve(base)
            if len(syms) > 1:                      # 别名命中多个基因：不猜，交人工对照表
                syms = set()
        if not syms:
            syms = resolve_tokens(base, idx, {})   # 'HER2/ERBB2'：各段并集
            if len(syms) > 1:
                syms = set()
        if syms:
            manning_canon[base] = syms
        else:
            pending.append((base, group, family, sub, prot))

    # 序列证据：未解析的 Manning 名，蛋白序列与"能按名称解析的 FASTA 名"完全相同
    seq2syms = defaultdict(set)
    for n, s in seqs.items():
        if len(s) >= 100:
            seq2syms[s] |= idx.resolve(n)
    for base, group, family, sub, prot in pending:
        syms = seq2syms.get(prot, set())
        if len(syms) == 1:
            manning_canon[base] = syms
    manning_unresolved = [base for base, *_ in entries if base not in manning_canon]

    kinome, family_of = set(), {}                  # family_of: 符号 -> (Group, Family, Subfamily)
    for base, group, family, sub, prot in entries:
        for c in manning_canon.get(base, ()):
            kinome.add(c)
            family_of[c] = (group, family, sub)
    # 人工对照表里带分类的行：补充 Manning 之后才命名 / 未收录的激酶
    for m in manual.values():
        syms, fam = manual_syms(m), upper(m.family)
        if syms and fam:
            for c in syms:
                kinome.add(c)
                family_of[c] = (upper(m.group), fam, upper(m.subfamily))

    # ---------- 原始名来源 ----------
    ks = pd.read_csv(KSR_CSV, dtype=str).fillna("")
    ks["K"] = ks["Kinase"].map(upper)
    ks["S"] = ks["EPSD_ID"].str.strip() + "_" + ks["Position"].str.strip()
    ks_u = ks.drop_duplicates(["K", "S"])
    n_edges = ks_u.groupby("K")["S"].nunique().to_dict()
    ppi = pd.read_csv(PPI_CSV, dtype=str).fillna("")
    ppi_names = set(ppi["preferredName_A"].map(upper)) | set(ppi["preferredName_B"].map(upper))
    raw_names = sorted(set(n_edges) | ppi_names | set(seqs))

    fetch_acc = {}
    if FETCH_REPORT.exists():
        fr = pd.read_csv(FETCH_REPORT, dtype=str).fillna("")
        fetch_acc = {upper(r.kinase): r.source.split(":")[1].strip()
                     for r in fr.itertuples(index=False) if r.status == "ok" and ":" in r.source}

    seq2names = defaultdict(list)
    for n, s in seqs.items():
        if len(s) >= 100:
            seq2names[s].append(n)

    # 忽略标点的键：现用符号 / 旧符号 / 别名 / Manning 名
    key2syms = defaultdict(set)
    for sym in idx.rows:
        if idx.is_protein_coding(sym):
            key2syms[compact(sym)].add(sym)
    for table in (idx.prev, idx.alias):
        for key, syms in table.items():
            key2syms[compact(key)] |= {c for c in syms if idx.is_protein_coding(c)}
    for mname, syms in manning_canon.items():
        key2syms[compact(mname)] |= syms

    def via_keys(name) -> set:
        """'PKC_ALPHA' -> PKCA、'CAM-KII_ALPHA' -> CAMK2A 这类写法的键匹配。"""
        key = compact(name)
        if key.startswith("CAMKII"):               # CaM-KII = CAMK2；'CAMK1A' 这类现用名不受影响
            key = "CAMK2" + key[6:]
        elif key.startswith("CAMKIV"):
            key = "CAMK4" + key[6:]
        elif key.startswith("CAMKI"):
            key = "CAMK1" + key[5:]
        if key in key2syms:
            return set(key2syms[key])
        for word, letter in GREEK.items():
            if key.endswith(word) and key[:-len(word)] + letter in key2syms:
                return set(key2syms[key[:-len(word)] + letter])
        return set()

    def resolve_any(x) -> set:
        return idx.resolve(x) or set(manning_canon.get(x, ())) or via_keys(x)

    def family_span(syms):
        """一组现用符号若同属一个 Manning 家族，返回该家族名；否则返回空串。"""
        fams = {family_of[c][1] for c in syms if c in family_of}
        return next(iter(fams)) if syms and len(fams) == 1 and "" not in fams else ""

    group_labels = manning_families | manning_groups | manning_subfams

    # ---------- 逐个原始名解析 ----------
    rec = {}                                       # 原始名（含物种后缀的原样）-> dict
    for raw in raw_names:
        out = dict(raw_name=raw, canonical="", entity_type="unmapped", match_source="",
                   family="", review=False, candidates="")
        m = manual.get(raw)
        if m is not None:                          # 0 人工对照
            syms, etype = manual_syms(m), upper(m.entity_type).lower()
            if not etype:
                etype = ("kinase" if next(iter(syms)) in kinome else "non_kinase") if len(syms) == 1 \
                    else ("group" if syms else "unmapped")
            out.update(canonical=next(iter(syms)) if len(syms) == 1 else "", entity_type=etype,
                       match_source="manual",
                       family=upper(m.family) or (family_span(syms) if etype == "group" else ""))
            rec[raw] = out
            continue
        base, species = strip_species(raw)
        if species and species != HUMAN_SUFFIX:    # 非人物种：不并入人激酶
            out.update(entity_type="non_human", match_source=f"species:{species}")
            rec[raw] = out
            continue
        name = base
        syms, src = idx.resolve(name), idx.source_of(name)          # 1-3 名称索引
        pm = PAREN_RE.match(name) if not syms else None             # 4 'ERK2(MAPK1)'
        if pm:
            syms = resolve_any(pm.group(2))
            if syms:
                src = "paren_symbol"
        im = ISO_RE.match(name) if not syms else None               # 4 'BRSK1 ISO2'
        if im:
            syms = resolve_any(im.group(1))
            if syms:
                src = "isoform"
        if not syms and GROUP_SUFFIX_RE.search(name):               # 5 'PKC_GROUP'
            out.update(entity_type="group", match_source="group_suffix")
            rec[raw] = out
            continue
        if not syms:                                                # 6 Manning 名
            syms = set(manning_canon.get(name, ()))
            if syms:
                src = "manning_name"
        if not syms and "/" in name:                                # 4 'PKG1/CGK-I'
            syms = resolve_tokens(name, idx, manning_canon)
            if syms:
                src = "token"
        if not syms:                                                # 7 忽略标点 / 希腊字母
            syms = via_keys(name)
            if syms:
                src = "compact_key"
        if len(syms) > 1 and len(syms & kinome) == 1:               # 别名命中多个基因，只有一个是激酶
            syms, src = syms & kinome, src + "+unique_kinase"
        if len(syms) == 1:
            sym = next(iter(syms))
            out.update(canonical=sym, match_source=src,
                       entity_type="kinase" if sym in kinome else "non_kinase")
            rec[raw] = out
            continue
        if len(syms) > 1:
            fam = family_span(syms & kinome)
            if len(syms & kinome) == len(syms) and fam:
                out.update(entity_type="group", family=fam, match_source=src)
            else:
                out.update(entity_type="ambiguous", review=True, match_source=src,
                           candidates=";".join(sorted(syms)))
            rec[raw] = out
            continue
        if name in group_labels:                                   # 8 家族 / 亚家族 / 大类名
            out.update(entity_type="group", match_source="family_name",
                       family=name if name in manning_families else "")
            rec[raw] = out
            continue
        acc = fetch_acc.get(name)                                  # 9 UniProt accession
        genes = (idx.uniprot.get(acc, set()) & kinome) if acc else set()
        if len(genes) == 1:
            out.update(canonical=next(iter(genes)), match_source="uniprot_acc", entity_type="kinase")
            rec[raw] = out
            continue
        out["review"] = raw in n_edges
        rec[raw] = out

    # ---------- 现用符号语义 ----------
    # 同一个字符串在 KinBase 命名与现用符号里可能是不同的蛋白（GCK=MAP4K2 / 葡糖激酶，
    # MST1=STK4 / 巨噬细胞刺激因子，PDK1=PDPK1 / 丙酮酸脱氢酶激酶…）。STRING 返回的
    # preferredName 和 UniProt 按基因名检索得到的序列名是现用符号语义，不能套用上面的
    # KinBase 对照规则，这里只按 HGNC 名称索引解析，单独给出三列（*_sym）。
    for n in raw_names:
        c, et, src = "", "unmapped", ""
        base, species = strip_species(n)
        if species and species != HUMAN_SUFFIX:
            et, src = "non_human", f"species:{species}"
        else:
            syms, s0 = idx.resolve(base), idx.source_of(base)
            if len(syms) > 1 and len(syms & kinome) == 1:
                syms, s0 = syms & kinome, s0 + "+unique_kinase"
            if len(syms) == 1:
                c = next(iter(syms))
                et, src = ("kinase" if c in kinome else "non_kinase"), s0
            elif len(syms) > 1:
                src = s0
                et = "group" if (len(syms & kinome) == len(syms) and family_span(syms)) else "ambiguous"
        rec[n].update(canonical_sym=c, entity_type_sym=et, match_source_sym=src)

    # ---------- 10 同序列证据：未解析的名字与已解析名字序列完全相同 ----------
    by_seq = defaultdict(set)
    for n in raw_names:
        s = seqs.get(n)
        if s and rec[n]["entity_type"] == "kinase" and rec[n]["canonical"]:
            by_seq[s].add(rec[n]["canonical"])
    n_seq_merged = 0
    for n in raw_names:
        if rec[n]["canonical"] or rec[n]["entity_type"] in ("non_kinase", "non_human", "other"):
            continue
        targets = by_seq.get(seqs.get(n), set())
        if len(targets) == 1:
            tgt = next(iter(targets))
            others = sorted(x for x in seq2names[seqs[n]] if rec[x]["canonical"] == tgt and x != n)
            rec[n].update(canonical=tgt, entity_type="kinase", review=False, candidates="",
                          match_source=f"same_sequence:{others[0] if others else ''}")
            n_seq_merged += 1

    # ---------- 输出 ----------
    alias = pd.DataFrame([rec[n] for n in raw_names])
    # 规范符号本身也入表（即使它从未作为原始名出现），保证查 info(规范名) 得到 (规范名, kinase)
    self_rows = [dict(raw_name=sym, canonical=sym, entity_type="kinase", match_source="symbol", family="",
                      canonical_sym=sym, entity_type_sym="kinase", match_source_sym="symbol",
                      n_edges=0, in_ppi=False, has_seq=False, review=False, candidates="")
                 for sym in sorted(kinome) if sym not in rec]
    alias = pd.concat([alias, pd.DataFrame(self_rows)], ignore_index=True)
    alias["n_edges"] = alias["raw_name"].map(lambda n: int(n_edges.get(n, 0)))
    alias["in_ppi"] = alias["raw_name"].isin(ppi_names)
    alias["has_seq"] = alias["raw_name"].isin(seqs)
    cols = ["raw_name", "canonical", "entity_type", "match_source", "family",
            "canonical_sym", "entity_type_sym", "match_source_sym",
            "n_edges", "in_ppi", "has_seq", "review", "candidates"]
    alias[cols].to_csv(out_alias, index=False)

    fam_rows = [dict(Name=sym, Group=family_of[sym][0], Family=family_of[sym][1], Subfamily=family_of[sym][2])
                for sym in sorted(kinome)]
    # Manning 名未能映射到现用符号的条目以原名入表，使仍以原名成节点的激酶也有家族标签
    fam_rows += [dict(Name=base, Group=group, Family=family, Subfamily=sub)
                 for base, group, family, sub, prot in entries
                 if base not in manning_canon and base not in family_of]
    pd.DataFrame(fam_rows).to_csv(out_family, index=False)

    # ---------- 报告 ----------
    print(f"HGNC 基因数 {len(idx.rows)} | Manning 人源非伪基因条目 {len(entries)}，"
          f"映射到现用符号的 {len(entries) - len(manning_unresolved)} 个")
    print(f"未能映射的 Manning 名 {len(manning_unresolved)} 个（以原名进家族表）:")
    print("  " + ", ".join(manning_unresolved))
    print(f"激酶组成员 {len(kinome)} 个")
    k = alias[alias.raw_name.isin(n_edges)]
    print(f"\nKSR 表激酶名 {len(n_edges)} 个，唯一边 {len(ks_u)}，按实体类型：")
    for et, sub in k.groupby("entity_type"):
        print(f"  {et:<11} 名字 {len(sub):>4} 个，边 {int(sub.n_edges.sum()):>6} 条")
    print("\n按匹配来源（仅 kinase 类型）:")
    kin = alias[alias.entity_type == "kinase"].copy()
    kin["source_kind"] = kin["match_source"].str.split(":").str[0]
    for src, sub in kin.groupby("source_kind"):
        print(f"  {src:<28} 名字 {len(sub):>4} 个，边 {int(sub.n_edges.sum()):>6} 条")

    def name_list(sub):
        return ", ".join(f"{r.raw_name}({r.n_edges})" for r in sub.itertuples(index=False))

    for et in ("non_kinase", "group", "other", "ambiguous"):
        sub = k[k.entity_type == et].sort_values("n_edges", ascending=False)
        if len(sub):
            print(f"\n[{et}] KSR 中的名字 {len(sub)} 个（共 {int(sub.n_edges.sum())} 条边）:")
            print("  " + name_list(sub))
    todo = k[k.entity_type == "unmapped"].sort_values("n_edges", ascending=False)
    print(f"\n[unmapped] KSR 中未解析的名字 {len(todo)} 个（共 {int(todo.n_edges.sum())} 条边）:")
    print("  " + name_list(todo))
    uniq = alias[alias.match_source.str.contains("unique_kinase")].sort_values("n_edges", ascending=False)
    print(f"\n别名命中多个基因、取其中唯一激酶的名字 {len(uniq)} 个（建议过目）:")
    print("  " + name_list(uniq))
    sym_names = fasta_names([UNIPROT_FASTA]) | ppi_names
    diff = alias[alias.raw_name.isin(sym_names)
                 & ((alias.canonical != alias.canonical_sym) | (alias.entity_type != alias.entity_type_sym))]
    print(f"\n两种语义下结论不同、且会出现在 STRING / UniProt 来源里的名字 {len(diff)} 个"
          f"（这些来源按现用符号语义处理）:")
    for r in diff.sort_values("n_edges", ascending=False).itertuples(index=False):
        print(f"  {r.raw_name}: KinBase 语义 -> {r.canonical or '-'}（{r.entity_type}） | "
              f"现用符号语义 -> {r.canonical_sym or '-'}（{r.entity_type_sym}）")

    merged = alias[(alias.entity_type == "kinase") & alias.canonical.ne("")].groupby("canonical").raw_name.apply(list)
    merged = merged[merged.map(len) > 1]
    print(f"\n合并了多个名字的激酶 {len(merged)} 个（括号为唯一边数；同序列补充解析 {n_seq_merged} 个名字）:")
    for canon, names in sorted(merged.items(), key=lambda kv: -sum(n_edges.get(n, 0) for n in kv[1])):
        print(f"  {canon:<10} <- " + ", ".join(f"{n}({n_edges.get(n, 0)})" for n in sorted(names)))
    print(f"\n已写出: {out_alias}\n已写出: {out_family}")


if __name__ == "__main__":
    main()
