"""
检索结果可视化生成器。

输入:
    - visualize/sim_matrix.txt      : 1000×1000 相似度矩阵（空格分隔）
    - visualize/MSRVTT_test.1000.csv: 查询元数据（key, vid_key, video_id, sentence）

输出:
    - visualize/retrieval.json      : 每条查询的Top-K检索结果（JSON数组）

格式说明:
    对第 i 条查询，取相似度矩阵第 i 行的值，
    按相似度从大到小排序，取前 K 条作为检索结果输出。
"""

import json
import csv
import numpy as np


def load_sim_matrix(path):
    """加载相似度矩阵（空格分隔的文本文件）。"""
    return np.loadtxt(path, dtype=np.float32)


def load_queries(path):
    """
    从 CSV 加载查询信息。

    CSV 列: key, vid_key, video_id, sentence
    """
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                'ret': row['key'],
                'video': row['video_id'],
                'query': row['sentence']
            })
    return queries


def generate_retrieval_results(sim_matrix_path, csv_path, output_path, topk=5):
    """
    生成检索结果 JSON。

    对每条查询（矩阵的每一行），找到与其相似度最高的 topk 条结果。
    """
    sim_matrix = load_sim_matrix(sim_matrix_path)
    queries = load_queries(csv_path)

    # 取前 1000 条，确保矩阵和数据对齐
    n = min(1000, len(queries), sim_matrix.shape[0])
    sim_matrix = sim_matrix[:n, :n]

    results = []
    for i in range(n):
        q = queries[i]
        row = sim_matrix[i]

        # 按相似度从大到小排序，取前 topk 个索引
        top_indices = np.argsort(row)[::-1][:topk]

        topk_results = []
        for rank, idx in enumerate(top_indices, start=1):
            candidate = queries[idx]
            topk_results.append({
                'rank': rank,
                'video': candidate['video'],
                'similarity': round(float(row[idx]), 3),
                'query_text': candidate['query']
            })

        # 检索成功 = 正确视频排在第 1 位（Top-1）
        is_success = topk_results[0]['video'] == q['video']

        results.append({
            'ret': q['ret'],
            'video': q['video'],
            'query': q['query'],
            'is_success': is_success,
            'results': topk_results
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已生成检索结果: {output_path}")
    print(f"  总查询数  : {n}")
    print(f"  每查询 Top: {topk}")

if __name__ == '__main__':
    # ==================== 用户可调参数 ====================
    SIM_MATRIX_PATH = 'visualize/sim_matrix.txt'
    CSV_PATH = 'visualize/MSRVTT_test.1000.csv'
    OUTPUT_PATH = 'visualize/retrieval.json'
    TOPK = 5                                  
    # =====================================================

    generate_retrieval_results(SIM_MATRIX_PATH, CSV_PATH, OUTPUT_PATH, TOPK)