#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为异构图的位点节点添加序列窗口(motif)特征。

输入 FASTA（origin_data/phosphosite_windows.fasta），表头格式:
    >EP0021246|S298|window_15
    AAETPPRPRTPGRPLSSYGMDSRPPMAIFEL
表头前两段以 '_' 拼接成位点名 EP0021246_S298，与 graph/build_joint_graph.py 中
site_id = EPSD_ID + '_' + Position 的节点命名一致。

FASTA 由激酶-底物关系表逐行展开生成，同一位点会重复出现：
取首次出现的窗口，同一站点若出现不同序列则告警。
窗口约定磷酸化残基位于 len(seq)//2 处；统一截取/补零到 2*flank+1
（默认 31，即 ±15）长度。

编码: 0 = 填充/非标准残基；1..20 = 标准氨基酸（'ACDEFGHIKLMNPQRSTVWY'）。
输出 data['site'].window（LongTensor [num_site, 2*flank+1]），
供 model.hgt_transformer.SiteMotifEncoder 编码后并入位点表示。

用法:
    python data_prep/add_motif_features.py \
        --graph origin_data/joint_hetero_graph.pt \
        --fasta origin_data/phosphosite_windows.fasta \
        --out origin_data/joint_hetero_graph_motif.pt
"""

import argparse
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
CENTRAL_RESIDUES = {'S', 'T', 'Y'}


def parse_fasta(path):
    """
    返回 (位点名 -> 首次出现的窗口序列, 总条目数, 序列不一致的位点数)。
    位点名取表头前两段以 '_' 拼接（>EP0021246|S298|window_15 -> EP0021246_S298）。
    """
    entries = {}          # name -> [seq, seq, ...]（同一位点每次出现记一条）
    name, chunks = None, []

    def flush():
        if name is not None:
            entries.setdefault(name, []).append(''.join(chunks))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                flush()
                parts = line[1:].split('|')
                name = parts[0] + '_' + parts[1]
                chunks = []
            elif line and name is not None:
                chunks.append(line)
        flush()

    total = sum(len(v) for v in entries.values())
    conflicts = sum(1 for v in entries.values() if len(set(v)) > 1)
    first = {n: v[0] for n, v in entries.items()}
    return first, total, conflicts


def encode_window(seq: str, flank: int) -> np.ndarray:
    """以 len(seq)//2 为中心截取 2*flank+1 长度；越界位置补 0，非标准字符记 0。"""
    length = 2 * flank + 1
    arr = np.zeros(length, dtype=np.int64)
    start = len(seq) // 2 - flank
    for i in range(length):
        j = start + i
        if 0 <= j < len(seq):
            arr[i] = AA_TO_ID.get(seq[j].upper(), 0)
    return arr


def main():
    parser = argparse.ArgumentParser(description="为位点节点添加序列窗口(motif)整数编码")
    parser.add_argument('--graph', required=True, help='异构图 .pt 路径')
    parser.add_argument('--fasta', required=True, help='位点窗口 FASTA 路径')
    parser.add_argument('--flank', type=int, default=15, help='窗口单侧长度（默认 15，即 31 聚体）')
    parser.add_argument('--out', required=True, help='输出图 .pt 路径')
    args = parser.parse_args()

    data = torch.load(args.graph, map_location='cpu', weights_only=False)
    site_names = list(data['site'].node_names)
    num_site = len(site_names)

    flank = args.flank
    fixed_len = 2 * flank + 1
    windows, total_entries, conflicts = parse_fasta(args.fasta)

    bad_center = sum(1 for s in windows.values() if s[len(s) // 2].upper() not in CENTRAL_RESIDUES)
    truncated = sum(1 for s in windows.values() if len(s) > fixed_len)
    padded = sum(1 for s in windows.values() if len(s) < fixed_len)
    if conflicts:
        logging.warning(f"{conflicts} 个位点的重复 FASTA 条目序列不一致，已取首次出现")
    if bad_center:
        logging.warning(f"{bad_center} 个窗口在中心位置(len//2)不是 S/T/Y，仍按该中心编码，建议上游重新生成")
    logging.info(
        f"FASTA: 共 {total_entries} 条记录, {len(windows)} 个唯一位点; "
        f"窗口统一为 {fixed_len} 长 (截断 {truncated}, 补零 {padded})"
    )

    aligned = np.zeros((num_site, fixed_len), dtype=np.int64)
    matched = 0
    missing_names = []
    for j, name in enumerate(site_names):
        seq = windows.get(str(name).strip())
        if seq is not None:
            aligned[j] = encode_window(seq, flank)
            matched += 1
        else:
            missing_names.append(name)
    if missing_names:
        logging.warning(f"{len(missing_names)} 个图位点在 FASTA 中未找到（保持全 0 编码），示例: {missing_names[:5]}")

    data['site'].window = torch.from_numpy(aligned)
    torch.save(data, args.out)
    logging.info(f"完成: 位点 {num_site} 个，匹配 {matched}；已写入 {args.out}")


if __name__ == '__main__':
    main()
