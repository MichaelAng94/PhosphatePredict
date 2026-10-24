# generate_kinase_fasta.py
# 用途：从 KinBase 分类文件（Excel）中提取激酶名称和序列，生成 FASTA 文件
# 输入：origin_data/kinase_family.xls（需包含 'Name' 和 'Protein' 列）
# 输出：origin_data/kinase_sequences.fasta

import os
import pandas as pd

# ==================== 配置 ====================
INPUT_EXCEL = 'origin_data/kinase_family.xls'
OUTPUT_FASTA = 'origin_data/kinase_sequences_out.fasta'

# 列名映射（如果实际列名不同，请修改此处）
COL_NAME = 'Name'      # 激酶名称列
COL_PROTEIN = 'Protein' # 蛋白质序列列

# ==================== 主程序 ====================
def main():
    if not os.path.exists(INPUT_EXCEL):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_EXCEL}")
    
    print(f"正在读取 {INPUT_EXCEL} ...")
    # 尝试读取 .xls 文件（支持 .xls 和 .xlsx）
    try:
        df = pd.read_excel(INPUT_EXCEL, engine='xlrd')
    except Exception as e:
        # 如果 xlrd 无法处理，尝试 openpyxl
        print(f"使用 xlrd 失败，尝试 openpyxl: {e}")
        df = pd.read_excel(INPUT_EXCEL, engine='openpyxl')
    
    # 检查必要列是否存在
    if COL_NAME not in df.columns:
        raise KeyError(f"文件中未找到列 '{COL_NAME}'，可用列: {list(df.columns)}")
    if COL_PROTEIN not in df.columns:
        raise KeyError(f"文件中未找到列 '{COL_PROTEIN}'，可用列: {list(df.columns)}")
    
    # 过滤掉序列缺失或为空的行
    df_valid = df[[COL_NAME, COL_PROTEIN]].dropna(subset=[COL_PROTEIN])
    df_valid = df_valid[df_valid[COL_PROTEIN].str.strip() != '']
    
    print(f"共找到 {len(df_valid)} 条有效激酶序列")
    # 读取 consolidated_all_kinases.csv，获取需要保留的激酶名称集合
    csv_path = 'origin_data/consolidated_all_kinases.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"激酶统计文件不存在: {csv_path}")

    df_kinase = pd.read_csv(csv_path)
    if 'Kinase' not in df_kinase.columns:
        raise KeyError(f"CSV 文件中缺少 'Kinase' 列，可用列: {list(df_kinase.columns)}")

    valid_kinases = set(df_kinase['Kinase'].astype(str).str.strip())

    # 根据激酶名称过滤
    # df_valid = df_valid[df_valid[COL_NAME].astype(str).str.strip().isin(valid_kinases)]

    df_out = df_valid[~df_valid[COL_NAME].astype(str).str.strip().isin(valid_kinases)]

    print(f"过滤后共保留 {len(df_valid)} 条激酶序列（仅限统计文件中存在的激酶）")
    
    # 写入 FASTA 文件
    with open(OUTPUT_FASTA, 'w') as f:
        for _, row in df_out.iterrows():
            name = str(row[COL_NAME]).strip()
            seq = str(row[COL_PROTEIN]).strip()
            # 去除序列中的空格和换行符（如果有）
            seq = seq.replace(' ', '').replace('\n', '').replace('\r', '')
            # 写入 FASTA 格式
            f.write(f">{name}\n")
            # 每行最多80个字符，美观起见可分行，但也可全部一行
            # 这里选择每80个字符换行
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')
    
    print(f"✅ FASTA 文件已生成: {OUTPUT_FASTA}")
    print(f"   序列数量: {len(df_out)}")

if __name__ == "__main__":
    main()