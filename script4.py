import os
import json
import argparse
from typing import List, Dict

# =============================
# IMPORT PIPELINE THẬT
# =============================
from script3 import ask          # hàm query GraphRAG của bạn
from graphrag import load_graph  # hàm load graph của bạn


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# SAVE / LOAD RESULT
# ============================================================

def save_result(results_dir: str, query_id: str, result: Dict):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{query_id}_result.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_result(results_dir: str, query_id: str) -> Dict:
    path = os.path.join(results_dir, f"{query_id}_result.json")

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# EXTRACT NODE ID THẬT (KHÔNG SINH GIẢ)
# ============================================================

def extract_node_ids(retrieved_nodes: list, k: int) -> List[str]:

    ids = []

    for node in retrieved_nodes[:k]:

        if isinstance(node, dict):

            if "node_id" in node:
                ids.append(str(node["node_id"]))

            elif "id" in node:
                ids.append(str(node["id"]))

        elif isinstance(node, str):
            ids.append(str(node))

    return ids


# ============================================================
# METRICS
# ============================================================

def compute_precision_recall_f1(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
):

    relevant_set = set(str(r) for r in relevant_ids)

    if not relevant_set:
        return 0.0, 0.0, 0.0

    retrieved_at_k = retrieved_ids[:k]

    tp = sum(1 for r in retrieved_at_k if r in relevant_set)

    precision = tp / k
    recall = tp / len(relevant_set)

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return precision, recall, f1


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
):

    relevant_set = set(str(r) for r in relevant_ids)

    for i, r in enumerate(retrieved_ids[:k]):
        if r in relevant_set:
            return 1 / (i + 1)

    return 0.0


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    dataset_path: str,
    graph_path: str,
    results_dir: str,
    k: int,
    generate: bool,
    eval_only: bool
):

    dataset = load_dataset(dataset_path)
    graph = load_graph(graph_path)

    all_metrics = []

    print("=" * 60)
    print("GraphRAG Generate + Evaluate (REAL NODE ID)")
    print("=" * 60)

    for sample in dataset:

        qid = sample["id"]
        question = sample["query"]
        relevant_ids = sample.get("relevant_node_ids", [])

        print(f"\n[{qid}] {question[:80]}")

        # ============================================
        # GENERATE (GỌI PIPELINE THẬT)
        # ============================================

        if generate or not eval_only:

            result = ask(question, graph, k=k)

            # result phải chứa:
            # {
            #   "retrieved_nodes": [
            #       {"node_id": ..., "score": ...}
            #   ]
            # }

            save_result(results_dir, qid, result)

        # ============================================
        # LOAD RESULT
        # ============================================

        result = load_result(results_dir, qid)

        if not result:
            print("  ❌ No result found")
            continue

        retrieved_nodes = result.get("retrieved_nodes", [])

        retrieved_ids = extract_node_ids(
            retrieved_nodes,
            k
        )

        # ============================================
        # EVALUATE
        # ============================================

        precision, recall, f1 = compute_precision_recall_f1(
            retrieved_ids,
            relevant_ids,
            k
        )

        mrr = compute_mrr(
            retrieved_ids,
            relevant_ids,
            k
        )

        print(
            f"  P={precision:.3f} "
            f"R={recall:.3f} "
            f"F1={f1:.3f} "
            f"MRR={mrr:.3f}"
        )

        all_metrics.append((precision, recall, f1, mrr))

    # ============================================
    # AVERAGE
    # ============================================

    if all_metrics:

        avg_p = sum(m[0] for m in all_metrics) / len(all_metrics)
        avg_r = sum(m[1] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m[2] for m in all_metrics) / len(all_metrics)
        avg_mrr = sum(m[3] for m in all_metrics) / len(all_metrics)

        print("\n" + "=" * 60)
        print("AVERAGE METRICS")
        print("=" * 60)

        print(f"Precision@{k}: {avg_p:.4f}")
        print(f"Recall@{k}:    {avg_r:.4f}")
        print(f"F1@{k}:        {avg_f1:.4f}")
        print(f"MRR@{k}:       {avg_mrr:.4f}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        graph_path=args.graph_path,
        results_dir=args.results_dir,
        k=args.k,
        generate=args.generate,
        eval_only=args.eval_only
    )