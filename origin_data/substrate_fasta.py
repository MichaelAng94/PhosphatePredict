# extract_phosphosite_windows.py
# 用途：从激酶-底物关系数据中提取位点周围 ±15 氨基酸的肽段窗口
# 输入：
#   1. origin_data/consolidated_kinase_substrate_pairs.csv（需包含 Substrate 和 Position 列）
#   2. origin_data/total.fasta（EPSD 提供的底物蛋白序列）
# 输出：origin_data/phosphosite_windows.fasta

import re
import os
import pandas as pd
from collections import defaultdict

# ==================== 配置 ====================
DATA_FILES = [
    ('PhosphoSitePlus', 'origin_data/PhosphoSitePlus.txt'),
    ('Phospho.ELM', 'origin_data/Phospho.ELM.txt'),
    ('PostMod', 'origin_data/PostMod.txt'),
    ('PSEA', 'origin_data/PSEA.txt'),
    ('RegPhos', 'origin_data/RegPhos.txt')
]
FASTA_FILE = 'origin_data/total.fasta'
OUTPUT_CSV = 'origin_data/kinase_substrate_pairs_detailed.csv'
OUTPUT_FASTA = 'origin_data/phosphosite_windows.fasta'
WINDOW = 15  # 左右各取15个氨基酸，总长度31

# ==================== 截取窗口 ====================
def parse_source(file_path, source_name):
    """返回 DataFrame，包含列：EPSD_ID, UniProt, Position, Kinase, Source"""
    records = []
    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return pd.DataFrame(columns=['EPSD_ID', 'UniProt', 'Position', 'Kinase', 'Source'])
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过表头（第一行）
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                # 如果列数不足，尝试按实际情况（至少需要 EPSD_ID, UniProt, Position, Kinase）
                continue
            epsd_id = parts[0].strip()
            uniprot = parts[1].strip()
            position = parts[2].strip()
            kinases_str = parts[3].strip()
            # 来源列（Source）是第5列，索引4，但为了安全，如果存在则使用
            # 这里我们直接使用 source_name 作为数据源标识，不依赖文件内的 Source 列
            # 但也可以保留，例如 parts[4]
            # 为统一，我们只记录 source_name
            # 拆分多个激酶
            for kinase_raw in kinases_str.split(';'):
                # kinase = normalize_kinase_name(kinase_raw)
                # if kinase:
                    records.append({
                        'EPSD_ID': epsd_id,
                        'UniProt': uniprot,
                        'Position': position,
                        'Kinase': kinase_raw,
                        'Source': source_name
                    })
        print(f"  ✓ {source_name}: 解析到 {len(records)} 条记录")
        return pd.DataFrame(records)
    except Exception as e:
        print(f"❌ 读取 {source_name} 时出错: {e}")
        return pd.DataFrame(columns=['EPSD_ID', 'UniProt', 'Position', 'Kinase', 'Source'])
    
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

def parse_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                parts = line[1:].split('|')
                current_id = parts[0]   # EPSD ID
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences

# ==================== 主程序 ====================
def main():
    all_dfs = []
    for source_name, file_path in DATA_FILES:
        df = parse_source(file_path, source_name)
        if not df.empty:
            all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 总共生成 {len(combined)} 条记录")
    print(f"   涉及激酶数量: {combined['Kinase'].nunique()}")
    print(f"   涉及底物数量: {combined['UniProt'].nunique()}")
    combined.to_csv(OUTPUT_CSV, index=False)

    # 1. 读取关系对，需要 Substrate (UniProt) 和 Position 列
    df_pairs = pd.read_csv(OUTPUT_CSV)
    
    unique_sites = df_pairs[['EPSD_ID', 'Position']]
    print(f"共有 {len(unique_sites)} 个 (EPSD_ID, 位点) 对")

    sequences = parse_fasta(FASTA_FILE)
    print(f"共加载 {len(sequences)} 个 EPSD 序列")

    # 2. 解析 FASTA
    print("正在解析 total.fasta ...")
    sequences = parse_fasta(FASTA_FILE)
    print(f"共加载 {len(sequences)} 个底物序列")

    # 3. 提取窗口并写入 FASTA
    count = 0
    with open(OUTPUT_FASTA, 'w') as f_out:
        for _, row in unique_sites.iterrows():
            epsd_id = row['EPSD_ID']
            pos_str = str(row['Position'])
            # 提取数字部分
            pos_num = int(re.search(r'\d+', pos_str).group())
            seq = sequences.get(epsd_id)
            window_seq = extract_window(seq, pos_num, WINDOW)
            header = f">{epsd_id}|{pos_str}|window_{WINDOW}"
            f_out.write(header + '\n')
            for i in range(0, len(window_seq), 80):
                f_out.write(window_seq[i:i+80] + '\n')
            count += 1

    print(f"✅ 成功提取 {count} 个位点窗口，保存至 {OUTPUT_FASTA}")

if __name__ == "__main__":
    main()
