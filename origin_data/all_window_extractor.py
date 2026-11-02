#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Total.fasta 中提取所有 S/T/Y 位点的窗口序列（左右各15），输出为 FASTA 文件。
窗口超出边界用 '-' 填充。
"""

import os

INPUT_FASTA = "origin_data/Total.fasta"
OUTPUT_FASTA = "origin_data/all_windows.fasta"
WINDOW = 15

def extract_window(sequence, position, window=15):
    """
    从序列中提取以 position（1-indexed）为中心的窗口。
    若超出边界，用 '-' 填充。
    """
    idx = position - 1  # 转为 0-indexed
    start = max(0, idx - window)
    end = min(len(sequence), idx + window + 1)
    extracted = sequence[start:end]
    # 填充左右缺失
    left_missing = window - (idx - start)
    right_missing = window - (end - (idx + 1))
    if left_missing > 0:
        extracted = '-' * left_missing + extracted
    if right_missing > 0:
        extracted = extracted + '-' * right_missing
    return extracted

def parse_protein_id(header):
    """
    从 FASTA 标题行提取蛋白质 ID（取第一个 '|' 前的部分）。
    例如 ">EP0000010|O00141|SGK1" -> "EP0000010"
    """
    if header.startswith('>'):
        header = header[1:]
    parts = header.split('|')
    return parts[0]

def main():
    seen_sequences = set()  # 记录已经写入的窗口序列字符串
    with open(INPUT_FASTA, 'r') as inf, open(OUTPUT_FASTA, 'w') as outf:
        current_id = None
        current_seq = []
        for line in inf:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # 保存上一条序列
                if current_id is not None and current_seq:
                    seq = ''.join(current_seq).upper()
                    for pos, aa in enumerate(seq, start=1):
                        if aa in ('S', 'T', 'Y'):
                            window_seq = extract_window(seq, pos, WINDOW)
                            if window_seq not in seen_sequences:
                                seen_sequences.add(window_seq)
                                fasta_id = f">{current_id}_{aa}{pos}"
                                outf.write(f"{fasta_id}\n{window_seq}\n")
                # 开始新序列
                current_id = parse_protein_id(line)
                current_seq = []
            else:
                current_seq.append(line)
        # 处理最后一条序列
        if current_id is not None and current_seq:
            seq = ''.join(current_seq).upper()
            for pos, aa in enumerate(seq, start=1):
                if aa in ('S', 'T', 'Y'):
                    window_seq = extract_window(seq, pos, WINDOW)
                    if window_seq not in seen_sequences:
                        seen_sequences.add(window_seq)
                        fasta_id = f">{current_id}_{aa}{pos}"
                        outf.write(f"{fasta_id}\n{window_seq}\n")
    print(f"去重后共保留 {len(seen_sequences)} 条唯一窗口序列，保存至 {OUTPUT_FASTA}")

if __name__ == "__main__":
    main()