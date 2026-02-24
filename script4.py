"""
script4_eval.py
---------------
Chạy GraphRAG pipeline (script3) cho từng sample trong ground truth dataset,
lưu result theo format chuẩn của framework, sau đó tính metrics.

Pipeline:
    data/ground_truth/dataset.json
            ↓
    script3 ask() → data/results/{query_id}_result.json
            ↓
    core_metrics.evaluate_single_sample()
            ↓
    reports/eval_{timestamp}.json  +  reports/charts/

Usage:
    python script4_eval.py --dataset data/ground_truth/dataset.json
    python script4_eval.py --dataset data/ground_truth/dataset.json --k 10
    python script4_eval.py --dataset data/ground_truth/dataset.json --setup
    python script4_eval.py --eval_only --results_dir data/results  # chỉ tính metrics, không chạy pipeline
"""

import os
import re
import json
import math
import argparse
import datetime
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent))
from script3 import ask, get_driver, setup_graph_algorithms

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

RESULTS_DIR = Path("data/results")
REPORTS_DIR = Path("reports")
for d in [RESULTS_DIR, REPORTS_DIR, REPORTS_DIR / "charts"]:
    d.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "precision":         0.7,
    "recall":            0.7,
    "f1":                0.65,
    "mrr":               0.6,
    "entity_coverage":   0.8,
    "path_relevance":    0.7,
    "comprehensiveness": 0.7,
    "faithfulness":      0.7,
}

METRIC_LABELS = {
    "precision":         "Precision@K",
    "recall":            "Recall@K",
    "f1":                "F1 Score",
    "mrr":               "MRR",
    "entity_coverage":   "Entity Coverage",
    "path_relevance":    "Path Relevance",
    "comprehensiveness": "Comprehensiveness",
    "faithfulness":      "Faithfulness",
}

COLORS = {
    "primary": "#2563EB",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger":  "#DC2626",
    "muted":   "#6B7280",
    "bg":      "#F8FAFC",
    "grid":    "#E2E8F0",
}


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: CHẠY PIPELINE VÀ LƯU RESULT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_for_sample(driver, ai_client: OpenAI, sample: dict) -> dict:
    """
    Gọi script3 ask(), map output sang format chuẩn của framework,
    lưu ra data/results/{query_id}_result.json.

    Format result chuẩn (theo pipeline_connector.py):
      query_id, query, generated_answer, retrieved_nodes,
      traversal_path, communities_covered, context_text
    """
    query_id = sample["id"]
    memory = {}  # evaluation mode → không dùng conversation memory
    qa = ask(driver, ai_client, sample["query"], memory=memory, query_id=query_id)

    # Build context_text từ retrieved_nodes
    context_text = "\n\n".join(
        n.get("content", "") for n in qa.get("retrieved_nodes", [])
    )

    result = {
        "query_id":            query_id,
        "query":               sample["query"],
        "generated_answer":    qa["generated_answer"],
        "retrieved_nodes":     qa["retrieved_nodes"],
        "traversal_path":      qa["traversal_path"],
        "communities_covered": [str(c) for c in qa.get("communities_covered", [])],
        "context_text":        context_text,
    }

    out_path = RESULTS_DIR / f"{query_id}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: METRICS  (theo core_metrics.py của framework)
# ══════════════════════════════════════════════════════════════════════════════

def compute_precision_recall_f1(retrieved_ids: list, relevant_ids: list, k: int) -> dict:
    retrieved_ids = retrieved_ids[:k]
    retrieved_set = set(retrieved_ids)
    relevant_set  = set(relevant_ids)

    if not retrieved_set and not relevant_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not retrieved_set or not relevant_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp        = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved_set)
    recall    = tp / len(relevant_set)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "true_positives":  tp,
        "retrieved_count": len(retrieved_set),
        "relevant_count":  len(relevant_set),
    }


def compute_mrr(retrieved_ids: list, relevant_ids: list) -> float:
    relevant_set = set(relevant_ids)
    for rank, nid in enumerate(retrieved_ids, 1):
        if nid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def compute_entity_coverage(gold_entities: list, retrieved_entities: list) -> dict:
    if not gold_entities:
        return {"entity_coverage": 1.0, "covered": [], "missing": []}
    gold_set      = {e.lower().strip() for e in gold_entities}
    retrieved_set = {e.lower().strip() for e in retrieved_entities}
    covered = gold_set & retrieved_set
    missing = gold_set - retrieved_set
    score   = len(covered) / len(gold_set)
    return {
        "entity_coverage":  round(score, 4),
        "covered_entities": list(covered),
        "missing_entities": list(missing),
        "threshold_ok":     score >= 0.8,
    }


def compute_path_relevance(traversal_path: list, gold_path: list) -> float | None:
    """
    Path Relevance: tỷ lệ gold_path entities xuất hiện trong traversal.
    Trả về None nếu gold_path rỗng (global query).
    """
    if not gold_path:
        return None
    retrieved_entities = set()
    for hop in traversal_path:
        retrieved_entities.add(hop.get("from", "").lower())
        retrieved_entities.add(hop.get("to",   "").lower())
    gold_set = {e.lower() for e in gold_path}
    overlap  = len(retrieved_entities & gold_set)
    return round(overlap / len(gold_set), 4)


def compute_path_depth(graph: nx.Graph | None,
                       gold_path: list, gold_hop_count: int) -> int | None:
    """Path Depth: số hop ngắn nhất trong graph từ đầu đến cuối gold_path."""
    if not graph or not gold_path or len(gold_path) < 2:
        return None
    src, tgt = gold_path[0], gold_path[-1]
    if src not in graph or tgt not in graph:
        return None
    try:
        path = nx.shortest_path(graph, src, tgt)
        return len(path) - 1
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def compute_comprehensiveness(generated_answer: str,
                               expected_communities: list) -> float | None:
    """
    Comprehensiveness: keyword matching — community keyword có trong answer không.
    Trả về None nếu expected_communities rỗng.
    """
    if not expected_communities:
        return None
    answer_lower = generated_answer.lower()
    covered = [
        c for c in expected_communities
        if c.replace("_", " ").lower().split()[0] in answer_lower
    ]
    return round(len(covered) / len(expected_communities), 4)


def compute_faithfulness(generated_answer: str, context_text: str) -> float:
    """
    Faithfulness heuristic: token + bigram overlap giữa answer và context.
    (Không cần LLM — dùng RAGAS nếu cần chính xác hơn)
    """
    if not context_text or not generated_answer:
        return 0.0

    def tokenize(text):
        words   = re.findall(r'\b\w+\b', text.lower())
        bigrams = {f"{a} {b}" for a, b in zip(words, words[1:])}
        return set(words) | bigrams

    answer_tokens  = tokenize(generated_answer)
    context_tokens = tokenize(context_text)
    if not answer_tokens:
        return 0.0
    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 4)


def evaluate_single_sample(sample: dict, result: dict,
                            graph: nx.Graph | None, k: int) -> dict:
    """
    Tính tất cả metrics cho 1 sample.
    Mapping theo framework:
        relevant_node_ids  ↔  retrieved_nodes[].node_id  → Precision/Recall/MRR
        gold_entities      ↔  retrieved_nodes[].entities → Entity Coverage
        gold_path          ↔  traversal_path             → Path Relevance
        gold_path + graph  ↔  gold_hop_count             → Path Depth
        expected_communities ↔ generated_answer          → Comprehensiveness
        ground_truth_answer  ↔ context_text              → Faithfulness
    """
    retrieved_ids = [n["node_id"] for n in result.get("retrieved_nodes", [])]
    retrieved_entities = []
    for n in result.get("retrieved_nodes", []):
        retrieved_entities.extend(n.get("entities", []))

    prf = compute_precision_recall_f1(
        retrieved_ids, sample.get("relevant_node_ids", []), k=k
    )
    ec  = compute_entity_coverage(
        sample.get("gold_entities", []), retrieved_entities
    )

    gold_path = sample.get("gold_path", [])

    return {
        "query_id":    sample["id"],
        "query":       sample["query"],
        "query_type":  sample.get("query_type", "unknown"),

        # Standard IR
        "precision":   prf["precision"],
        "recall":      prf["recall"],
        "f1":          prf["f1"],
        "mrr":         compute_mrr(retrieved_ids, sample.get("relevant_node_ids", [])),

        # Graph-specific
        "entity_coverage": ec["entity_coverage"],

        # Multi-hop
        "path_relevance": compute_path_relevance(
            result.get("traversal_path", []), gold_path
        ),
        "path_depth": compute_path_depth(
            graph, gold_path, sample.get("gold_hop_count", 0)
        ),

        # Global
        "comprehensiveness": compute_comprehensiveness(
            result.get("generated_answer", ""),
            sample.get("expected_communities", [])
        ),

        # Faithfulness (heuristic)
        "faithfulness": compute_faithfulness(
            result.get("generated_answer", ""),
            result.get("context_text", "")
        ),

        # Debug info
        "entity_coverage_detail": ec,
        "prf_detail": prf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════

def safe_mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate_metrics(all_metrics: list) -> dict:
    metric_keys = ["precision", "recall", "f1", "mrr", "entity_coverage",
                   "path_relevance", "path_depth", "comprehensiveness", "faithfulness"]

    overall = {k: safe_mean([m.get(k) for m in all_metrics]) for k in metric_keys}
    overall["sample_count"] = len(all_metrics)

    query_types = set(m.get("query_type", "unknown") for m in all_metrics)
    by_type = {}
    for qt in sorted(query_types):
        qt_ms = [m for m in all_metrics if m.get("query_type") == qt]
        by_type[qt] = {k: safe_mean([m.get(k) for m in qt_ms]) for k in metric_keys}
        by_type[qt]["sample_count"] = len(qt_ms)

    return {"overall": overall, "by_query_type": by_type}


def print_summary(aggregate: dict, k: int):
    overall = aggregate["overall"]
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"\nPASS/FAIL vs Thresholds (K={k}):")
    for key, label in METRIC_LABELS.items():
        val    = overall.get(key)
        thresh = THRESHOLDS.get(key, 0.7)
        if val is None:
            print(f"  --  {label:25s}  N/A")
        elif val >= thresh:
            print(f"  OK  {label:25s}  {val:.4f}  (>= {thresh})")
        else:
            print(f"  XX  {label:25s}  {val:.4f}  (<  {thresh})")

    print(f"\n  Total samples: {overall['sample_count']}")

    by_type = aggregate.get("by_query_type", {})
    if by_type:
        print("\nBY QUERY TYPE:")
        for qt, m in by_type.items():
            print(f"\n  [{qt}]  n={m['sample_count']}")
            for key in ["precision", "recall", "f1", "entity_coverage", "faithfulness"]:
                val = m.get(key)
                if val is not None:
                    print(f"    {METRIC_LABELS.get(key, key):20s}: {val:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: VISUALIZE  (theo visualize.py của framework)
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(aggregate: dict, output_path: str):
    metrics_to_show = ["precision", "recall", "f1",
                       "entity_coverage", "faithfulness", "path_relevance"]
    overall = aggregate.get("overall", {})

    values = []
    labels = []
    for m in metrics_to_show:
        v = overall.get(m)
        if v is not None:
            values.append(v)
            labels.append(METRIC_LABELS.get(m, m))

    if not values:
        return

    N      = len(values)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]
    thresh_plot = [THRESHOLDS.get(metrics_to_show[i], 0.7)
                   for i in range(N)] + [THRESHOLDS.get(metrics_to_show[0], 0.7)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.fill(angles, thresh_plot, alpha=0.1, color=COLORS["warning"])
    ax.plot(angles, thresh_plot, color=COLORS["warning"], linewidth=1.5,
            linestyle="--", label="Threshold")
    ax.fill(angles, values_plot, alpha=0.25, color=COLORS["primary"])
    ax.plot(angles, values_plot, color=COLORS["primary"], linewidth=2.5,
            marker="o", markersize=8, label="GraphRAG")

    for angle, val, m in zip(angles[:-1], values, metrics_to_show):
        color = COLORS["success"] if val >= THRESHOLDS.get(m, 0.7) else COLORS["danger"]
        ax.annotate(f"{val:.2f}", xy=(angle, val), fontsize=10, fontweight="bold",
                    ha="center", va="center", color=color,
                    xytext=(0, 15), textcoords="offset points")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color=COLORS["muted"])
    ax.grid(color=COLORS["grid"], linewidth=1)
    plt.title("GraphRAG Evaluation — Metric Overview",
              size=14, fontweight="bold", pad=30, color="#1E293B")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  radar_overview.png → {output_path}")


def plot_bar_by_query_type(aggregate: dict, output_path: str):
    by_type = aggregate.get("by_query_type", {})
    if not by_type:
        return

    query_types   = list(by_type.keys())
    metrics       = ["precision", "recall", "f1", "entity_coverage"]
    metric_labels = [METRIC_LABELS[m] for m in metrics]
    x             = np.arange(len(metrics))
    width         = 0.8 / max(len(query_types), 1)
    colors_list   = [COLORS["primary"], COLORS["success"],
                     COLORS["warning"], COLORS["danger"]]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for i, (qtype, color) in enumerate(zip(query_types, colors_list)):
        vals   = [by_type[qtype].get(m) or 0 for m in metrics]
        offset = (i - len(query_types) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=qtype.replace("_", " ").title(),
                        color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.axhline(y=0.7, color=COLORS["warning"], linestyle="--",
               linewidth=1.5, alpha=0.8, label="Threshold (0.7)")
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Metrics by Query Type", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  bar_by_type.png    → {output_path}")


def plot_scatter_pr(per_sample: list, output_path: str):
    if not per_sample:
        return

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    type_colors = {
        "single_hop": COLORS["primary"],
        "multi_hop":  COLORS["success"],
        "global":     COLORS["warning"],
        "unknown":    COLORS["muted"],
    }

    for m in per_sample:
        p     = m.get("precision", 0)
        r     = m.get("recall",    0)
        qtype = m.get("query_type", "unknown")
        color = type_colors.get(qtype, COLORS["muted"])
        ax.scatter(r, p, color=color, s=120, alpha=0.8,
                   edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(m["query_id"], (r, p), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=COLORS["muted"])

    ax.axhline(y=0.7, color=COLORS["warning"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=0.7, color=COLORS["warning"], linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between([0.7, 1.0], 0.7, 1.0, alpha=0.05, color=COLORS["success"])
    ax.text(0.85, 0.92, "Target Zone", color=COLORS["success"], fontsize=9,
            ha="center", transform=ax.transAxes, alpha=0.8)

    patches = [
        mpatches.Patch(color=c, label=t.replace("_", " ").title())
        for t, c in type_colors.items()
        if any(m.get("query_type") == t for m in per_sample)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    ax.set_xlabel("Recall@K", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision@K", fontsize=12, fontweight="bold")
    ax.set_title("Precision vs Recall per Sample", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(color=COLORS["grid"], linewidth=1, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  scatter_pr.png     → {output_path}")


def plot_summary_table(aggregate: dict, output_path: str):
    overall = aggregate.get("overall", {})
    rows    = []
    for key, label in METRIC_LABELS.items():
        val    = overall.get(key)
        thresh = THRESHOLDS.get(key, 0.7)
        if val is not None:
            rows.append([label, f"{val:.3f}", f"{thresh}", "PASS" if val >= thresh else "FAIL"])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    table = ax.table(
        cellText=rows, colLabels=["Metric", "Score", "Threshold", "Status"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for j in range(4):
        table[(0, j)].set_facecolor(COLORS["primary"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, 1):
        for j in range(4):
            table[(i, j)].set_facecolor("#FFFFFF" if i % 2 == 0 else COLORS["bg"])
        is_pass = row[3] == "PASS"
        table[(i, 3)].set_facecolor(
            COLORS["success"] + "22" if is_pass else COLORS["danger"] + "22"
        )
        table[(i, 3)].set_text_props(
            color=COLORS["success"] if is_pass else COLORS["danger"],
            fontweight="bold"
        )

    ax.set_title("GraphRAG Evaluation Summary",
                 fontsize=14, fontweight="bold", pad=15, color="#1E293B")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  summary_table.png  → {output_path}")


def generate_charts(aggregate: dict, per_sample: list, charts_dir: Path):
    print("\nGenerating charts...")
    plot_radar_chart(aggregate,         str(charts_dir / "radar_overview.png"))
    plot_bar_by_query_type(aggregate,   str(charts_dir / "bar_by_type.png"))
    plot_scatter_pr(per_sample,         str(charts_dir / "scatter_pr.png"))
    plot_summary_table(aggregate,       str(charts_dir / "summary_table.png"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_path: str) -> list:
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    # Hỗ trợ cả "id" và "query_id"
    for s in samples:
        if "id" not in s and "query_id" in s:
            s["id"] = s["query_id"]
    return samples


def run_eval(
    dataset_path: str,
    k: int = 5,
    graph_path: str | None = None,
    eval_only: bool = False,
    results_dir: str = "data/results",
):
    print(f"\n{'='*60}")
    print("GraphRAG Evaluation")
    print(f"{'='*60}")

    samples = load_dataset(dataset_path)
    print(f"Dataset : {dataset_path}  ({len(samples)} samples)")

    # Load NetworkX graph nếu có (cho Path Depth metric)
    graph = None
    if graph_path and Path(graph_path).exists():
        with open(graph_path, encoding="utf-8") as f:
            gdata = json.load(f)
        graph = nx.Graph()
        for n in gdata.get("nodes", []):
            graph.add_node(n if isinstance(n, str) else n.get("id", ""))
        for e in gdata.get("edges", []):
            graph.add_edge(
                e.get("source", e.get("from", "")),
                e.get("target", e.get("to", "")),
                relation=e.get("relation", e.get("type", ""))
            )
        print(f"Graph   : {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ── Chạy pipeline hoặc load results đã có ────────────────────────────────
    all_results: dict[str, dict] = {}

    if eval_only:
        print(f"\nLoading existing results from {results_dir}/ ...")
        for fpath in Path(results_dir).glob("*_result.json"):
            with open(fpath, encoding="utf-8") as f:
                r = json.load(f)
            all_results[r["query_id"]] = r
        print(f"  Loaded {len(all_results)} result files")
    else:
        driver    = get_driver()
        ai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"\nRunning pipeline for {len(samples)} samples (K={k})...\n")

        for i, sample in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {sample['id']} "
                  f"({sample.get('query_type','?')}) — {sample['query'][:60]}...")
            try:
                result = run_pipeline_for_sample(driver, ai_client, sample)
                all_results[sample["id"]] = result
                print(f"  → saved: {RESULTS_DIR}/{sample['id']}_result.json")
            except Exception as e:
                print(f"  ERROR: {e}")

        driver.close()

    # ── Tính metrics ──────────────────────────────────────────────────────────
    print(f"\nComputing metrics (K={k})...\n")
    all_metrics = []
    missing     = []

    for sample in samples:
        qid = sample["id"]
        if qid not in all_results:
            missing.append(qid)
            result = {
                "query_id": qid, "query": sample["query"],
                "generated_answer": "", "retrieved_nodes": [],
                "traversal_path": [], "communities_covered": [], "context_text": "",
            }
        else:
            result = all_results[qid]

        metrics = evaluate_single_sample(sample, result, graph=graph, k=k)
        all_metrics.append(metrics)

        m = metrics
        print(f"  [{qid}] {sample['query'][:55]}...")
        print(f"    P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  MRR={m['mrr']:.3f}  "
              f"EC={m['entity_coverage']:.3f}  Faith={m['faithfulness']:.3f}", end="")
        if m.get("path_relevance") is not None:
            print(f"  PathRel={m['path_relevance']:.3f}", end="")
        if m.get("comprehensiveness") is not None:
            print(f"  Comp={m['comprehensiveness']:.3f}", end="")
        print()

    # ── Aggregate + print ─────────────────────────────────────────────────────
    aggregate = aggregate_metrics(all_metrics)
    print_summary(aggregate, k)
    if missing:
        print(f"\n  WARNING: Missing results for: {missing}")

    # ── Lưu report ────────────────────────────────────────────────────────────
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "config": {
            "dataset":     dataset_path,
            "k":           k,
            "timestamp":   datetime.datetime.now().isoformat(),
            "eval_only":   eval_only,
        },
        "aggregate":       aggregate,
        "per_sample":      all_metrics,
        "missing_results": missing,
    }
    report_path = REPORTS_DIR / f"eval_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Charts ────────────────────────────────────────────────────────────────
    generate_charts(aggregate, all_metrics, REPORTS_DIR / "charts")

    print(f"\n{'='*60}")
    print(f"Results  → {RESULTS_DIR}/  ({len(all_results)} files)")
    print(f"Report   → {report_path}")
    print(f"Charts   → {REPORTS_DIR}/charts/")
    print(f"{'='*60}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphRAG Evaluation — chạy pipeline + tính metrics + vẽ charts"
    )
    parser.add_argument("--dataset",
                        default="data/ground_truth/dataset.json",
                        help="Path tới ground truth dataset.json")
    parser.add_argument("--k",
                        type=int, default=5,
                        help="Top-K cho Precision/Recall (default: 5)")
    parser.add_argument("--graph_path",
                        default=None,
                        help="Path tới graph.json (optional, cho Path Depth metric)")
    parser.add_argument("--eval_only",
                        action="store_true",
                        help="Chỉ tính metrics từ results đã có, không chạy pipeline")
    parser.add_argument("--results_dir",
                        default="data/results",
                        help="Thư mục chứa result files (dùng với --eval_only)")
    parser.add_argument("--setup",
                        action="store_true",
                        help="Chạy Community Detection + PageRank trước khi eval")
    args = parser.parse_args()

    if args.setup:
        driver = get_driver()
        setup_graph_algorithms(driver)
        driver.close()

    run_eval(
        dataset_path=args.dataset,
        k=args.k,
        graph_path=args.graph_path,
        eval_only=args.eval_only,
        results_dir=args.results_dir,
    )