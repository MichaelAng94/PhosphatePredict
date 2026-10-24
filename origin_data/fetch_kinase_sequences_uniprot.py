# fetch_kinase_sequences_uniprot.py
# 用途：为图节点中缺失 ESM 特征的激酶（零向量激酶）从 UniProt 拉取蛋白序列
#
# 背景：
#   build_joint_graph.py 的候选激酶来自 底物关系 CSV ∪ PPI CSV（大写归一化）。
#   其中在两个已有 fasta（kinase_sequences.fasta / kinase_sequences_out.fasta）
#   中没有序列的激酶，最终只靠 Node2Vec 进入图，esm_x 为零向量（上次运行 208 个）。
#
# 本脚本流程：
#   1. 重算"缺失序列的图节点激酶"清单（与 build_joint_graph.py 口径一致）
#   2. 优先用 mapping_epsd_to_gene.csv 的 gene_symbol -> uniprot_id 定位 accession
#      （一个基因多个 accession 时逐一尝试，取第一个成功返回的）
#   3. 映射不到的激酶，退回 UniProt 在线检索（人类 + 已评审）
#   4. 结果写入 origin_data/kinase_sequences_uniprot.fasta，头部为图归一化激酶名
#   5. 成功/失败清单写入 origin_data/fetch_report.csv 便于人工跟进
#
# 运行（需要联网，约 200 次请求，含限速约 3~5 分钟）：
#   cd origin_data && python3 fetch_kinase_sequences_uniprot.py
#
# 依赖：仅 Python 标准库（urllib / csv），无需 torch。

import csv
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# ==================== 输入输出 ====================
KS_CSV = os.path.join(HERE, 'kinase_substrate_pairs_detailed.csv')   # 激酶-底物关系
PPI_CSV = os.path.join(HERE, '..', 'PPI', 'kinase_ppi_network.csv')  # 激酶 PPI
MAPPING_CSV = os.path.join(HERE, 'mapping_epsd_to_gene.csv')         # EPSD -> 基因/UniProt
EXISTING_FASTA = [                                                  # 已有序列（有则跳过）
    os.path.join(HERE, 'kinase_sequences.fasta'),
    os.path.join(HERE, 'kinase_sequences_out.fasta'),
]
OUT_FASTA = os.path.join(HERE, 'kinase_sequences_uniprot.fasta')     # 新序列输出
OUT_REPORT = os.path.join(HERE, 'fetch_report.csv')                   # 成功/失败报告

UNIPROT_ENTRY = 'https://rest.uniprot.org/uniprotkb/{acc}.fasta'
UNIPROT_SEARCH = ('https://rest.uniprot.org/uniprotkb/search?'
                  'query={query}&format=fasta')

REQUEST_GAP = 1.0     # 请求间隔（秒），对 UniProt 保持礼貌
MAX_RETRY = 3
TIMEOUT = 30
MIN_LEN = 50           # 序列最短长度（激酶域都远大于此，过短说明拉错条目）
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')  # 只保留标准氨基酸，忽略 U/X/B/Z 等
ORGANISM_HUMAN = '9606'


def norm(name: str) -> str:
    """与 build_joint_graph.py 的 normalize_kinase_name 完全一致"""
    return name.strip().upper() if isinstance(name, str) else ''


def http_get(url: str) -> str:
    """带重试的 GET，返回响应文本；失败返回 None"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PhosphatePredict-seq-fetch/1.0'})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode('utf-8')
        except Exception as e:
            if attempt == MAX_RETRY:
                print(f'  [重试耗尽] {url} -> {e}')
                return None
            time.sleep(2 * attempt)
    return None


def parse_fasta_entries(text: str):
    """解析 fasta 文本为 [(header, sequence), ...]"""
    entries, header, seq = [], None, []
    for line in text.splitlines():
        if line.startswith('>'):
            if header is not None:
                entries.append((header, ''.join(seq)))
            header, seq = line[1:], []
        else:
            seq.append(line.strip())
    if header is not None:
        entries.append((header, ''.join(seq)))
    return entries


def clean_sequence(seq: str) -> str:
    """只保留标准氨基酸"""
    return ''.join(c for c in seq.upper() if c in VALID_AA)


# ==================== 第 1 步：重算缺失序列的图节点激酶 ====================
def build_target_kinases():
    """与 build_joint_graph.py 同口径：
       候选 = KS ∪ PPI；无 ESM 的激酶需在 PPI 图中才能得到 n2v 后备并成为图节点；
       目标 = 这些图节点候选中在已有 fasta 里没有序列的激酶。"""
    ks_kinases = set()
    with open(KS_CSV, newline='') as f:
        for row in csv.DictReader(f):
            ks_kinases.add(norm(row['Kinase']))

    ppi_nodes = set()
    with open(PPI_CSV, newline='') as f:
        for row in csv.DictReader(f):
            ppi_nodes.add(norm(row.get('preferredName_A', '')))
            ppi_nodes.add(norm(row.get('preferredName_B', '')))
    ppi_nodes.discard('')

    candidates = ks_kinases | ppi_nodes          # 全部候选（830 个）
    graph_candidates = candidates & ppi_nodes     # 无序列也能进图的（靠 PPI/n2v）

    have_seq = set()
    for path in EXISTING_FASTA:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.startswith('>'):
                    have_seq.add(norm(line[1:].split()[0]))

    targets = sorted(graph_candidates - have_seq)
    print(f'候选激酶: {len(candidates)} | 图节点候选: {len(graph_candidates)} | '
          f'已有序列: {len(graph_candidates & have_seq)} | 缺序列待拉取: {len(targets)}')
    return targets


# ==================== 第 2 步：gene_symbol -> uniprot accession 映射 ====================
def build_gene_to_uniprots():
    """从 mapping_epsd_to_gene.csv 建 {大写基因名: [uniprot_id, ...]}；
       同一基因的多个 accession 按出现顺序去重保留。"""
    gene_map = {}
    with open(MAPPING_CSV, newline='') as f:
        for row in csv.DictReader(f):
            gene = norm(row['gene_symbol'])
            acc = (row.get('uniprot_id') or '').strip()
            if gene and acc and acc not in gene_map.setdefault(gene, []):
                gene_map[gene].append(acc)
    return gene_map


# ==================== 第 3 步：两种方式拉取序列 ====================
def fetch_by_accession(acc: str):
    """直接按 accession 拉取，返回 (序列, accession) 或 None"""
    text = http_get(UNIPROT_ENTRY.format(acc=urllib.parse.quote(acc)))
    if not text:
        return None
    entries = parse_fasta_entries(text)
    if not entries:
        return None
    seq = clean_sequence(entries[0][1])
    if len(seq) < MIN_LEN:
        print(f'  [丢弃] {acc}: 序列过短({len(seq)}aa)，疑似错误条目')
        return None
    return seq, acc


def fetch_by_gene_search(kinase: str):
    """UniProt 在线检索兜底：人类 + 已评审，基因名精确匹配"""
    query = f'gene_exact:{kinase} AND organism_id:{ORGANISM_HUMAN} AND reviewed:true'
    url = UNIPROT_SEARCH.format(query=urllib.parse.quote(query))
    text = http_get(url)
    if not text:
        return None
    for header, raw_seq in parse_fasta_entries(text):
        seq = clean_sequence(raw_seq)
        if len(seq) < MIN_LEN:
            continue
        # header 形如 'sp|P42345|MTOR_HUMAN ...'，取 accession 供记录
        parts = header.split('|')
        acc = parts[1] if len(parts) >= 2 else header.split()[0]
        return seq, acc
    return None


def main():
    targets = build_target_kinases()
    gene_map = build_gene_to_uniprots()

    results, failures = [], []
    for i, k in enumerate(targets, 1):
        got, source = None, ''

        # 方式一：映射文件 gene_symbol -> uniprot_id（上次核对可覆盖 136/208）
        for acc in gene_map.get(k, []):
            got = fetch_by_accession(acc)
            if got:
                source = f'mapping:{got[1]}'
                break

        # 方式二：UniProt 检索兜底（如 MTOR/PRKACA 等未入映射的）
        if not got:
            got = fetch_by_gene_search(k)
            if got:
                source = f'search:{got[1]}'

        if got:
            results.append((k, got[0], source))
            print(f'[{i}/{len(targets)}] {k}: OK ({source}, {len(got[0])}aa)')
        else:
            failures.append(k)
            print(f'[{i}/{len(targets)}] {k}: 未获取到（映射与检索均失败，需人工核对别名）')

        time.sleep(REQUEST_GAP)

    # ==================== 第 4 步：写出 fasta 与报告 ====================
    with open(OUT_FASTA, 'w') as f:
        for k, seq, _ in results:
            f.write(f'>{k}\n')
            for j in range(0, len(seq), 60):
                f.write(seq[j:j + 60] + '\n')

    with open(OUT_REPORT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kinase', 'status', 'source', 'seq_len'])
        for k, seq, src in results:
            w.writerow([k, 'ok', src, len(seq)])
        for k in failures:
            w.writerow([k, 'failed', '', ''])

    print(f'\n完成: 成功 {len(results)} / {len(targets)}，失败 {len(failures)}')
    print(f'序列输出: {OUT_FASTA}')
    print(f'报告输出: {OUT_REPORT}')
    if failures:
        print('失败激酶（多为旧别名/融合基因/组名，需人工建立别名表后补充）:')
        print('  ' + ', '.join(failures))


if __name__ == '__main__':
    main()
