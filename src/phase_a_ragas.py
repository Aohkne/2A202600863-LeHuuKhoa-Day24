from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    overall = {
        metric: sum(getattr(r, metric) for r in results) / len(results)
        for metric in DIAGNOSTIC_TREE
    } if results else {}

    report = {
        "total_questions": len(results),
        "overall": overall,
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        # Bonus Phase A: adversarial phải khó hơn factual thì test set mới phân biệt được
        "adversarial_harder_than_factual": (
            per_dist.get("adversarial", {}).get("avg_score", 1.0)
            < per_dist.get("factual", {}).get("avg_score", 0.0)
        ),
        "bottom_10": bottom_10(results),
        "per_question": [
            {"question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "answer": r.answer,
             "faithfulness": round(r.faithfulness, 4),
             "answer_relevancy": round(r.answer_relevancy, 4),
             "context_precision": round(r.context_precision, 4),
             "context_recall": round(r.context_recall, 4),
             "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        groups[item["distribution"]].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Dùng lại `evaluate_ragas()` của Day 18 (src/m4_eval.py) để cùng một cấu hình
    judge LLM + embeddings được áp dụng cho cả hai lab, thay vì gọi thẳng
    ragas.evaluate() với default OpenAI models.
    """
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:  # khi chạy `python src/phase_a_ragas.py`
        from m4_eval import evaluate_ragas

    result = evaluate_ragas(
        questions=[a["question"] for a in answers],
        answers=[a["answer"] for a in answers],
        contexts=[a["contexts"] for a in answers],
        ground_truths=[a["ground_truth"] for a in answers],
    )
    per_question = result.get("per_question", [])
    if not per_question:
        print("⚠️  evaluate_ragas() không trả về per_question — kiểm tra API key / RAGAS.")
        return []

    return [
        RagasResult(
            question_id=a["id"], distribution=a["distribution"],
            question=a["question"], answer=a["answer"],
            contexts=a["contexts"], ground_truth=a["ground_truth"],
            faithfulness=r.faithfulness,
            answer_relevancy=r.answer_relevancy,
            context_precision=r.context_precision,
            context_recall=r.context_recall,
        )
        for a, r in zip(answers, per_question)
    ]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất."""
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    bottom = sorted_asc[:10]
    output = []
    for i, r in enumerate(bottom):
        diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank":          i + 1,
            "question_id":   r.question_id,
            "distribution":  r.distribution,
            "question":      r.question,
            "avg_score":     round(r.avg_score, 4),
            "worst_metric":  r.worst_metric,
            "diagnosis":     diag,
            "suggested_fix": fix,
        })
    return output


FAILURE_THRESHOLD = 0.6  # worst_metric dưới ngưỡng này mới tính là failure thực sự

DISTRIBUTIONS = ["factual", "multi_hop", "adversarial"]


def cluster_analysis(results: list[RagasResult], threshold: float = FAILURE_THRESHOLD) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Hai ma trận 4×3:
      * `matrix`         — mọi câu hỏi, đếm theo worst_metric (mỗi câu đúng 1 ô).
      * `failure_matrix` — chỉ những câu có điểm worst_metric < threshold.

    Dominant distribution được chọn theo *tỷ lệ* failure, không theo số đếm thô:
    mỗi câu hỏi đóng góp đúng một ô vào `matrix`, nên đếm thô chỉ phản ánh
    distribution nào có nhiều câu hỏi nhất (20/20/10) chứ không phải nơi pipeline yếu.
    """
    matrix = {metric: {d: 0 for d in DISTRIBUTIONS} for metric in DIAGNOSTIC_TREE}
    failure_matrix = {metric: {d: 0 for d in DISTRIBUTIONS} for metric in DIAGNOSTIC_TREE}

    for r in results:
        matrix[r.worst_metric][r.distribution] += 1
        if getattr(r, r.worst_metric) < threshold:
            failure_matrix[r.worst_metric][r.distribution] += 1

    per_dist_total = {d: sum(1 for r in results if r.distribution == d) for d in DISTRIBUTIONS}
    failure_rate = {
        d: round(sum(failure_matrix[m][d] for m in failure_matrix) / per_dist_total[d], 3)
        if per_dist_total[d] else 0.0
        for d in DISTRIBUTIONS
    }
    avg_by_dist = {
        d: round(sum(r.avg_score for r in results if r.distribution == d) / per_dist_total[d], 4)
        if per_dist_total[d] else 0.0
        for d in DISTRIBUTIONS
    }

    dominant_dist   = max(DISTRIBUTIONS, key=lambda d: failure_rate[d])
    dominant_metric = max(failure_matrix, key=lambda m: sum(failure_matrix[m].values())) \
        if any(sum(v.values()) for v in failure_matrix.values()) \
        else max(matrix, key=lambda m: sum(matrix[m].values()))
    worst_cell = max(
        ((m, d, failure_matrix[m][d]) for m in failure_matrix for d in DISTRIBUTIONS),
        key=lambda x: x[2],
    )

    insight = (
        f"Distribution '{dominant_dist}' có failure rate cao nhất "
        f"({failure_rate[dominant_dist]:.0%}, avg_score {avg_by_dist[dominant_dist]:.3f}). "
        f"Metric '{dominant_metric}' là điểm yếu chủ đạo — cluster nặng nhất là "
        f"({worst_cell[0]} × {worst_cell[1]}) với {worst_cell[2]} câu dưới ngưỡng {threshold}. "
        f"Chẩn đoán: {DIAGNOSTIC_TREE[dominant_metric][0]}. "
        f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}"
    )
    return {
        "threshold": threshold,
        "matrix": matrix,
        "failure_matrix": failure_matrix,
        "failure_rate_by_distribution": failure_rate,
        "avg_score_by_distribution": avg_by_dist,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "worst_cluster": {"metric": worst_cell[0], "distribution": worst_cell[1],
                          "count": worst_cell[2]},
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print("\nAvg score theo distribution:")
        for dist, avg in clusters["avg_score_by_distribution"].items():
            print(f"  {dist:<12} avg={avg:.3f}  failure_rate="
                  f"{clusters['failure_rate_by_distribution'][dist]:.0%}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
        print(f"Insight: {clusters['insight']}")

        adv = clusters["avg_score_by_distribution"]["adversarial"]
        fac = clusters["avg_score_by_distribution"]["factual"]
        print(f"\nAdversarial ({adv:.3f}) < factual ({fac:.3f})? "
              f"{'✓ có — test set phân biệt được độ khó' if adv < fac else '✗ không'}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
