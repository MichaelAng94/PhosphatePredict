#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""激酶命名归一（合并别名）与家族表加载——建图脚本与训练脚本共用的唯一入口。

规范名 = HGNC 现用基因符号（比较大写）。原始名到规范名的映射表由
data_prep/build_kinase_alias_map.py 生成（origin_data/kinase_alias_map.csv），
可人工编辑后直接重跑下游；本模块只负责查表，不依赖 HGNC / xlrd。

两种命名语义——同一个字符串在不同来源里可能是不同的蛋白：
    'kinbase'  KSR 表，以及按 KinBase / Manning 命名取序列的文件
               （kinase_sequences.fasta、kinase_sequences_out.fasta）。
               例：GCK=MAP4K2、MST1=STK4、PDK1=PDPK1。
    'symbol'   STRING 返回的 preferredName，以及 UniProt 按基因名检索得到的序列
               （kinase_sequences_uniprot.fasta）。
               例：GCK=葡糖激酶、MST1=巨噬细胞刺激因子、PDK1=丙酮酸脱氢酶激酶 1。
调用方必须按名字的来源选择语义，不能对所有来源套用同一列。

kinase_alias_map.csv 列:
    raw_name          原始名（大写，与建图/训练的归一口径一致）
    canonical         规范名（kinbase 语义）；泛称/歧义/未解析时为空
    entity_type       kinase | group | non_kinase | non_human | other | ambiguous | unmapped
    canonical_sym     规范名（现用符号语义）
    entity_type_sym   实体类型（现用符号语义）
    match_source      匹配依据
    family            泛称所指的 Manning 家族（仅 group 有值）
    review            True 表示需人工核对
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIAS_MAP_PATH = str(PROJECT_ROOT / "origin_data/kinase_alias_map.csv")
FAMILY_TABLE_PATH = str(PROJECT_ROOT / "origin_data/kinase_family.csv")

SEMANTICS = ("kinbase", "symbol")

# 建图时排除的实体类型：不是"某一个具体的人蛋白激酶"，其边无法归属到规范激酶。
# ambiguous / unmapped 不排除——保持与旧图一致（按原名成节点），只是标记出来待核对。
EXCLUDED_ENTITY_TYPES = ("group", "non_kinase", "non_human", "other")


def upper_name(raw) -> str:
    """与建图/训练脚本一致的名字归一：去空白、转大写；非字符串返回空串。"""
    return raw.strip().upper() if isinstance(raw, str) else ""


class KinaseNamer:
    """原始激酶名 -> (规范名, 实体类型) 的查表器。"""

    def __init__(self, alias_map: pd.DataFrame):
        df = alias_map.copy()
        df["raw_name"] = df["raw_name"].map(upper_name)
        self.has_symbol_semantics = {"canonical_sym", "entity_type_sym"} <= set(df.columns)
        if self.has_symbol_semantics:
            df["canonical_sym"] = df["canonical_sym"].fillna("").astype(str).str.strip().str.upper()
            df["entity_type_sym"] = df["entity_type_sym"].fillna("unmapped").astype(str).str.strip()
        df["canonical"] = df["canonical"].fillna("").astype(str).str.strip().str.upper()
        df["entity_type"] = df["entity_type"].fillna("unmapped").astype(str).str.strip()
        self.table = df[df["raw_name"] != ""].drop_duplicates("raw_name").set_index("raw_name")

    @classmethod
    def load(cls, path: str = ALIAS_MAP_PATH) -> "KinaseNamer":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到别名映射表 {path}；先运行 python3 data_prep/build_kinase_alias_map.py 生成"
            )
        return cls(pd.read_csv(path, dtype=str))

    def info(self, raw, semantics: str = "kinbase") -> tuple:
        """返回 (规范名, 实体类型)。表中没有的名字原样返回大写名，类型 unmapped。

        semantics: 'kinbase'（KSR 表 / KinBase 命名的序列名）或 'symbol'（STRING / UniProt 按基因名
        检索得到的名字），见模块说明。表里没有 *_sym 列时 'symbol' 退回 'kinbase' 的结果。
        """
        if semantics not in SEMANTICS:
            raise ValueError(f"semantics 只能是 {SEMANTICS}，收到 {semantics!r}")
        name = upper_name(raw)
        if name and name in self.table.index:
            row = self.table.loc[name]
            if semantics == "symbol" and self.has_symbol_semantics:
                return (row["canonical_sym"] or name), row["entity_type_sym"]
            return (row["canonical"] or name), row["entity_type"]
        return name, "unmapped"

    def canonical(self, raw, semantics: str = "kinbase") -> str:
        return self.info(raw, semantics)[0]

    def entity_type(self, raw, semantics: str = "kinbase") -> str:
        return self.info(raw, semantics)[1]

    def aliases(self, canonical, semantics: str = "kinbase") -> list:
        """映射到该规范名的全部原始名（含规范名自身），按字母排序。"""
        col = "canonical_sym" if (semantics == "symbol" and self.has_symbol_semantics) else "canonical"
        rows = self.table[self.table[col] == upper_name(canonical)]
        return sorted(set(rows.index) | {upper_name(canonical)})


def load_family_table(path: str = FAMILY_TABLE_PATH) -> pd.DataFrame:
    """
    读取激酶家族表，统一返回含 Name(大写)/Group/Family/Subfamily 列的 DataFrame。
    支持新表 kinase_family.csv（规范名 + Manning 分类）与旧表 kinase_family.xls（需 xlrd）。
    """
    if str(path).lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, dtype=str).fillna("")
    if "Name" not in df.columns or "Family" not in df.columns:
        raise ValueError(f"家族表缺少 Name/Family 列: {path}，实际列: {list(df.columns)}")
    out = pd.DataFrame({"Name": df["Name"].map(upper_name)})
    for col in ("Group", "Family", "Subfamily"):
        out[col] = (df[col].astype(str).str.strip() if col in df.columns else "")
    return out.drop_duplicates("Name")
