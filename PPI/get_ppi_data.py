from pathlib import Path

import requests
import pandas as pd
import time
from typing import List

# ==================== 配置参数 ====================
SPECIES_ID = 9606              # 物种 ID，9606 代表人类 (Homo sapiens)
REQUIRED_SCORE = 0             # 最小得分阈值 (0-1000)，0 代表获取所有结果
REQUEST_DELAY = 1              # API 请求间隔（秒），避免请求过于频繁
BATCH_SIZE = 100               # API 单次查询允许的最大蛋白数量
# 以脚本位置定位项目根目录，从任意目录运行脚本时数据路径依然正确
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = str(PROJECT_ROOT / "origin_data/consolidated_all_kinases.csv")
# 输出固定到 PPI/ 目录，与 generate_ppi_graph.py、graph/build_joint_graph.py 的读取路径一致
OUTPUT_CSV = str(PROJECT_ROOT / "PPI/kinase_ppi_network.csv")

# ==================== 核心函数 ====================
def read_kinases(csv_path: str) -> List[str]:
    """
    从 CSV 文件中读取激酶列表。
    前提：CSV 文件必须包含 'Kinase' 列。
    """
    try:
        df = pd.read_csv(csv_path)
        if 'Kinase' not in df.columns:
            raise ValueError("CSV 文件缺少 'Kinase' 列，请检查文件格式。")
        kinases = df['Kinase'].dropna().unique().tolist()
        print(f"[✓] 从文件中成功加载 {len(kinases)} 个激酶。")
        return kinases
    except FileNotFoundError:
        print(f"[✗] 错误：文件 {csv_path} 未找到，请检查路径。")
        exit(1)
    except Exception as e:
        print(f"[✗] 读取 CSV 文件时出错：{e}")
        exit(1)

def fetch_ppi_network(proteins: List[str], species: int) -> List[dict]:
    """
    调用 STRING API 的 /network 端点，查询一组蛋白之间的所有相互作用。
    API 文档: https://string-db.org/help/api/
    """
    # 用 '\r' 分隔多个蛋白 ID
    identifiers = "\r".join(proteins)
    params = {
        "identifiers": identifiers,
        "species": species,
        "network_type": "functional",
        "required_score": REQUIRED_SCORE,
        "add_nodes": 0                     # 只返回输入列表内的互作关系
    }
    url = "https://string-db.org/api/json/network"

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()        # 如果返回了 HTTP 错误码，则触发异常
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[✗] API 请求失败：{e}")
        return []

# ==================== 主程序 ====================
def main():
    print("=== 激酶 PPI 网络获取工具 ===")
    kinases = read_kinases(INPUT_CSV)

    if not kinases:
        print("[✗] 未找到任何激酶数据，程序退出。")
        return

    # 由于 STRING API 单次查询有蛋白数量限制，如果激酶总数超过 BATCH_SIZE 则分批处理
    all_interactions = []
    total_batches = (len(kinases) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(kinases), BATCH_SIZE):
        batch = kinases[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\n[>] 正在处理第 {batch_num}/{total_batches} 批（包含 {len(batch)} 个激酶）...")

        interactions = fetch_ppi_network(batch, SPECIES_ID)

        if interactions:
            all_interactions.extend(interactions)
            print(f"[✓] 第 {batch_num} 批获取到 {len(interactions)} 条相互作用。")
        else:
            print(f"[!] 第 {batch_num} 批未获取到相互作用。")

        # 礼貌性地延迟，避免 API 限流
        if i + BATCH_SIZE < len(kinases):
            print(f"[*] 等待 {REQUEST_DELAY} 秒后继续...")
            time.sleep(REQUEST_DELAY)

    # 保存结果
    if all_interactions:
        # 将 API 返回的 JSON 数据转换为 DataFrame，并精简列
        df = pd.DataFrame(all_interactions)
        df.to_csv(str(PROJECT_ROOT / "PPI/raw_ppi_data.csv"), index=False)  # 保存原始数据以供调试
        # 保留最关键的三列：蛋白A、蛋白B、综合得分
        result_df = df[['preferredName_A', 'preferredName_B', 'score']]
        result_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[✓] 成功！已保存 {len(result_df)} 条激酶间 PPI 数据到文件：{OUTPUT_CSV}")
    else:
        print("[✗] 警告：未获取到任何 PPI 数据。请检查：")
        print("  1. 激酶名称是否为标准基因名或 UniProt ID")
        print("  2. 物种 ID 是否正确（当前为 9606）")
        print("  3. 网络连接是否正常")

if __name__ == "__main__":
    main()